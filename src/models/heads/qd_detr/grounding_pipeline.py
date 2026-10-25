import torch
from torch import nn
from transformers import AutoModel


class GroundingTextEncoder(nn.Module):
    """Token-level text encoder used by the grounding head."""

    def __init__(self, model_name: str, trainable: bool = True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        for parameter in self.model.parameters():
            parameter.requires_grad = trainable

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state


class FourDGroundingModel(nn.Module):
    """Attach QD-DETR temporal grounding to unpooled 4D encoder tokens."""

    def __init__(
        self,
        encoder: nn.Module,
        grounding_head: nn.Module,
        text_encoder: GroundingTextEncoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.grounding_head = grounding_head
        self.text_encoder = text_encoder

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    def forward(
        self,
        motion: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        video_padding_mask: torch.Tensor = None,
    ):
        if not hasattr(self.encoder, "encode_motion"):
            raise AttributeError(
                "The configured 4D encoder must expose an encode_motion method."
            )
        video_tokens = self.encoder.encode_motion(motion)
        text_tokens = self.text_encoder(input_ids, attention_mask)
        if video_padding_mask is None:
            video_mask = torch.ones(
                video_tokens.shape[:2], dtype=torch.bool, device=video_tokens.device
            )
        else:
            if video_padding_mask.shape[1] != video_tokens.shape[1]:
                raise ValueError(
                    "Video padding mask length must match the encoder token length."
                )
            video_mask = ~video_padding_mask.bool()
        return self.grounding_head(
            src_txt=text_tokens,
            src_txt_mask=attention_mask.bool(),
            src_vid=video_tokens,
            src_vid_mask=video_mask,
        )
