#!/usr/bin/env python3

import argparse
import importlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src/models/heads"))

from src.models.heads.qd_detr.model import build_model
from src.models.heads.qd_detr.grounding_pipeline import (
    FourDGroundingModel,
    GroundingTextEncoder,
)
from src.utils.train_utils import (
    ConfigManager,
    create_training_state_dict,
    setup_training_environment,
)


def namespace_from_config(config):
    return SimpleNamespace(**config)


def load_4d_encoder(config, device):
    encoder_config = ConfigManager.load_config(config["model"]["encoder_config_path"])
    pipeline_config = encoder_config["model"]
    module = importlib.import_module(pipeline_config["pipeline_module"])
    encoder_class = getattr(module, pipeline_config["pipeline_class"])
    encoder_args = dict(pipeline_config.get("pipeline_args", {}))
    if "feature_mode" in inspect.signature(encoder_class).parameters:
        encoder_args["feature_mode"] = "all"
    wrapper_module_name = config["model"].get("encoder_wrapper_module")
    wrapper_class_name = config["model"].get("encoder_wrapper_class")
    if wrapper_module_name and wrapper_class_name:
        wrapper_module = importlib.import_module(wrapper_module_name)
        encoder_class = getattr(wrapper_module, wrapper_class_name)
    encoder = encoder_class(**encoder_args).to(device)

    checkpoint_path = config["model"].get("encoder_checkpoint_path")
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        state = state.get("model_state_dict", state.get("model", state))
        
        
        # --- ADD THIS BLOCK --- Fix this later
        # Map 'fc_norm' from the 'avg' pretrained checkpoint to 'norm' expected by 'all' mode
        if "motion_encoder.model.fc_norm.weight" in state:
            state["motion_encoder.model.norm.weight"] = state.pop("motion_encoder.model.fc_norm.weight")
            state["motion_encoder.model.norm.bias"] = state.pop("motion_encoder.model.fc_norm.bias")
        # ----------------------
        
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"4D encoder checkpoint mismatch: missing={missing}, "
                f"unexpected={unexpected}"
            )
    return encoder, encoder_config, checkpoint if checkpoint_path else None


def build_grounding_model(config, device):
    grounding_config = config["model"]["grounding"]
    encoder, encoder_config, encoder_checkpoint = load_4d_encoder(config, device)
    encoder_args = encoder_config["model"].get("pipeline_args", {})
    encoder_dim = grounding_config.get("encoder_output_dim")
    for key in ("output_dim", "feature_dim", "embedding_dim", "motion_embedding_dims"):
        if encoder_dim is None:
            encoder_dim = encoder_args.get(key)
    if encoder_dim is None:
        for key in ("output_dim", "feature_dim", "embedding_dim", "num_features"):
            encoder_dim = getattr(encoder, key, None)
            if encoder_dim is not None:
                break
    if encoder_dim is None:
        raise ValueError(
            "Set model.grounding.encoder_output_dim when the configured "
            "4D encoder output dimension cannot be inferred."
        )
    data_config = config["data"]
    encoder_frames = data_config.get("sampled_frames", 160)
    if "dataset_config_path" in data_config:
        dataset_config = ConfigManager.load_config(data_config["dataset_config_path"])
        encoder_frames = dataset_config.get("sampled_frames", encoder_frames)
    if hasattr(encoder, "output_tokens_for_frames"):
        qd_args_video_tokens = encoder.output_tokens_for_frames(encoder_frames)
    else:
        qd_args_video_tokens = grounding_config.get("max_v_l")
    if qd_args_video_tokens is None:
        raise ValueError("Unable to determine the grounding video token count.")
    text_model_name = grounding_config["text_model_name"]
    text_encoder = GroundingTextEncoder(
        text_model_name, trainable=not grounding_config["freeze_text_encoder"]
    )
    if grounding_config.get("load_text_encoder_from_checkpoint", True):
        if encoder_checkpoint is None:
            raise ValueError(
                "load_text_encoder_from_checkpoint is enabled, but "
                "encoder_checkpoint_path is not configured."
            )
        checkpoint_state = encoder_checkpoint.get("state_dict", encoder_checkpoint)
        text_state = checkpoint_state.get("text_encoder_state_dict")
        if text_state is None:
            raise KeyError(
                "The encoder checkpoint does not contain "
                "'state_dict.text_encoder_state_dict'."
            )
        missing, unexpected = text_encoder.load_state_dict(text_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Text encoder checkpoint mismatch: missing={missing}, "
                f"unexpected={unexpected}"
            )

    qd_args = dict(grounding_config)
    qd_args.update(
        {
            "device": str(device),
            "dset_name": "hl",
            "a_feat_dir": None,
            "t_feat_dim": text_encoder.text_model.config.hidden_size,
            "v_feat_dim": encoder_dim,
            "max_v_l": qd_args_video_tokens,
            "aux_loss": grounding_config.get("aux_loss", True),
        }
    )
    # QD-DETR's transformer builder expects this CLI-style option even when
    # the grounding YAML omits it.
    qd_args.setdefault("pre_norm", False)
    grounding_head, criterion = build_model(namespace_from_config(qd_args))
    model = FourDGroundingModel(encoder, grounding_head, text_encoder).to(device)
    return model, criterion, AutoTokenizer.from_pretrained(text_model_name)


def build_dataset(config, split):
    data_config = config["data"]
    
    # Import class and collate function
    module = importlib.import_module(data_config["dataset_module"])
    dataset_class = getattr(module, data_config["dataset_class"])
    collate_fn = getattr(module, data_config["collate_fn"])

    # If external path is provided, load it; otherwise extract from data_config
    dataset_config_path = data_config.get("dataset_config_path")
    if dataset_config_path:
        dataset_config = ConfigManager.load_config(dataset_config_path)
    else:
        # Filter out dataloader-specific arguments
        excluded_keys = {
            "dataset_module", "dataset_class", "collate_fn", 
            "train_split", "val_split", "batch_size", 
            "num_workers", "pin_memory", "sequence_config"
        }
        dataset_config = {k: v for k, v in data_config.items() if k not in excluded_keys}
        
        # Inject seed if available
        if "sequence_config" in data_config and "seed" in data_config["sequence_config"]:
            dataset_config["seed"] = data_config["sequence_config"]["seed"]

    # Assign split and route the correct JSON file path
    dataset_config["split"] = split
    dataset_config["stitched_json_path"] = dataset_config["train_json"] if split == "train" else dataset_config["test_json"]

    # Pass the compiled dictionary to the dataset constructor
    return dataset_class(dataset_config), collate_fn

def make_optimizer(model, config):
    opt_config = config["training"]["optimizer"]
    parameter_groups = [
        {
            "params": list(model.encoder.parameters()),
            "lr": opt_config["encoder_lr"],
        },
        {
            "params": [p for p in model.grounding_head.parameters() if p.requires_grad],
            "lr": opt_config["grounding_lr"],
        },
        {
            "params": [p for p in model.text_encoder.parameters() if p.requires_grad],
            "lr": opt_config["text_lr"],
        },
    ]
    parameter_groups = [group for group in parameter_groups if group["params"]]
    if not parameter_groups:
        raise ValueError("No trainable parameters were found.")
    return AdamW(
        parameter_groups,
        weight_decay=opt_config.get("weight_decay", 1e-4),
        betas=tuple(opt_config.get("betas", [0.9, 0.999])),
        eps=opt_config.get("eps", 1e-8),
    )


def prepare_batch(batch, tokenizer, device, max_q_l):
    motion, queries, targets, video_padding_mask = batch
    tokens = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=max_q_l,
        return_tensors="pt",
    )
    
    # QD-DETR expects a single dictionary where keys map to a list of target dicts for the batch
    formatted_targets = {
        "span_labels": [
            {"spans": t["span_labels"][0]["spans"].to(device)} for t in targets
        ]
    }
    
    return (
        {
            "motion": motion.to(device, non_blocking=True),
            "input_ids": tokens["input_ids"].to(device),
            "attention_mask": tokens["attention_mask"].to(device),
            "video_padding_mask": video_padding_mask.to(device, non_blocking=True),
        },
        formatted_targets
    )


def run_epoch(model, criterion, loader, optimizer, tokenizer, config, device, train):
    model.train(train)
    criterion.train(train)
    total_loss = 0.0
    for batch in tqdm(loader, desc="train" if train else "val"):
        inputs, targets = prepare_batch(
            batch, tokenizer, device, config["model"]["grounding"]["max_q_l"]
        )
        with torch.set_grad_enabled(train):
            outputs = model(**inputs)
            loss_dict = criterion(outputs, targets)
            losses = sum(
                loss_dict[key] * criterion.weight_dict[key]
                for key in loss_dict
                if key in criterion.weight_dict
            )
            if train:
                optimizer.zero_grad()
                losses.backward()
                clip = config["training"].get("gradient_clipping")
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
        total_loss += losses.detach().item()
    return total_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config", default="configs/grounding_config.yaml")
    args = parser.parse_args()
    config = ConfigManager.load_config(args.config)
    device = torch.device(config["model"].get("device", "cuda"))
    logger, tb_logger, checkpoint_manager = setup_training_environment(
        config, config["experiment"]["name"]
    )

    train_dataset, collate_fn = build_dataset(config, config["data"]["train_split"])
    val_dataset, _ = build_dataset(config, config["data"]["val_split"])
    
    # create subset of the training dataset for debugging purposes
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset, range(min(128, len(train_dataset))))
    val_dataset = Subset(val_dataset, range(min(128, len(val_dataset))))
    loader_args = {
        "batch_size": config["data"]["batch_size"],
        "num_workers": config["data"]["num_workers"],
        "pin_memory": config["data"]["pin_memory"],
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    model, criterion, tokenizer = build_grounding_model(config, device)
    freeze = config["model"].get("freeze_encoder", False)
    warmup_freeze = config["model"].get("freeze_encoder_during_warmup", False)
    model.set_encoder_trainable(not (freeze or warmup_freeze))
    optimizer = make_optimizer(model, config)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["num_epochs"],
        eta_min=config["training"]["scheduler"].get("eta_min", 0.0),
    )

    best_loss = float("inf")
    for epoch in range(config["training"]["num_epochs"]):
        if warmup_freeze and epoch == config["model"].get("warmup_epochs", 0):
            model.set_encoder_trainable(True)
        train_loss = run_epoch(
            model, criterion, train_loader, optimizer, tokenizer, config, device, True
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model, criterion, val_loader, optimizer, tokenizer, config, device, False
            )
        scheduler.step()
        logger.info(f"Epoch {epoch + 1}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}")
        tb_logger.log_training_metrics(
            epoch + 1, train_loss, {"val_loss": val_loss}, optimizer.param_groups[0]["lr"]
        )
        state = create_training_state_dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            additional_state={"train_loss": train_loss, "val_loss": val_loss},
        )
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_manager.save_checkpoint(state, epoch, val_loss, is_best=True)
        checkpoint_manager.save_checkpoint(state, epoch, val_loss)


if __name__ == "__main__":
    main()
