import os
from typing import Optional, Tuple

import torch
from torch import nn

from transformers import AltCLIPModel, AltCLIPProcessor, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from model.tbps_model import CLIP


class AltCLIPVisionWrapper(nn.Module):
    def __init__(self, pretrained: AltCLIPModel):
        super().__init__()
        self.vision_model = pretrained.vision_model
        self.visual_projection = pretrained.visual_projection

    @property
    def dtype(self):
        return self.visual_projection.weight.dtype

    def forward(self, pixel_values: torch.Tensor, return_dense: bool = False, return_feature: bool = False):
        outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)
        seq = outputs.last_hidden_state
        projected_seq = self.visual_projection(seq)
        pooled = projected_seq[:, 0, :]

        if return_dense:
            return pooled, projected_seq
        if return_feature:
            return projected_seq
        return pooled


class AltCLIPTextWrapper(nn.Module):
    def __init__(self, pretrained: AltCLIPModel):
        super().__init__()
        self.text_model = pretrained.text_model
        self.text_projection = pretrained.text_projection

    @property
    def dtype(self):
        return self.text_projection.weight.dtype

    def forward(self, tokens, return_dense: bool = False):
        if isinstance(tokens, tuple):
            tokens = tokens[0]
        attention_mask = None
        if isinstance(tokens, BatchEncoding):
            attention_mask = tokens.attention_mask if "attention_mask" in tokens else None
            input_ids = tokens.input_ids
        elif isinstance(tokens, dict):
            attention_mask = tokens["attention_mask"] if "attention_mask" in tokens else None
            input_ids = tokens["input_ids"]
        else:
            input_ids = tokens
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        seq = outputs.last_hidden_state
        projected_seq = self.text_projection(seq)
        pooled = self.text_projection(outputs.pooler_output)

        if return_dense:
            return pooled, projected_seq
        return pooled


class AltCLIPTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, texts, context_length: int, mask_type: Optional[str] = None):
        if mask_type is not None:
            raise NotImplementedError("AltCLIP tokenizer does not support masked language modeling.")
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=context_length,
            return_tensors="pt",
        )
        return encoded


def build_altclip_clip(config, model_name: str = "BAAI/AltCLIP", device: Optional[str] = None) -> Tuple[CLIP, AltCLIPTokenizer, AltCLIPProcessor]:
    os.environ.setdefault("HF_HUB_DISABLE_AUTO_CONVERSION", "1")
    pretrained = AltCLIPModel.from_pretrained(model_name, use_safetensors=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = AltCLIPProcessor.from_pretrained(model_name)

    vision = AltCLIPVisionWrapper(pretrained)
    text = AltCLIPTextWrapper(pretrained)

    tokenizer_wrapper = AltCLIPTokenizer(tokenizer)
    num_classes = getattr(config.model, "num_classes", 1)
    clip_model = CLIP(config, vision, text, num_classes=num_classes, eps=config.experiment.ritc_eps,
                     tokenizer=tokenizer_wrapper)
    clip_model.logit_scale.data = pretrained.logit_scale.data.clone()

    if device is not None:
        clip_model.to(device)

    return clip_model, tokenizer_wrapper, processor
