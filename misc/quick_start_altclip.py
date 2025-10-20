"""Quick start script for running AltCLIP through the TBPS-CLIP training wrapper.

Now supports passing a local AltCLIP directory via --altclip_pretrained.
"""

import os
import pathlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from easydict import EasyDict
from PIL import Image
from transformers import AutoTokenizer
import argparse

os.environ.setdefault("HF_HUB_DISABLE_AUTO_CONVERSION", "1")

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model import tbps_model
from model.altclip_adapter import build_altclip_clip


class _DummyEDA:
    def random_deletion(self, sentence, alpha):
        return sentence


tbps_model.EDA = _DummyEDA


def build_config(device: str, vocab_size: int) -> EasyDict:
    return EasyDict({
        "device": device,
        "misc": EasyDict({"seed": 42}),
        "experiment": EasyDict({
            "input_resolution": [224, 224],
            "text_length": 77,
            "dropout": 0.0,
            "eda_alpha": 0.0,
            "back_trans": False,
            "backtrans_p": 0.0,
            "mixgen": False,
            "mixgen_type": "cat",
            "mixgen_p": 0.0,
            "mixgen_ratio": 0.0,
            "mvs_image": False,
            "nitc_ratio": 1.0,
            "ss": False,
            "ss_ratio": 0.0,
            "citc": True,
            "citc_lambda1": 0.25,
            "citc_lambda2": 0.25,
            "citc_ratio": 0.1,
            "ritc": True,
            "ritc_eps": 1.0e-2,
            "ritc_ratio": 1.0,
            "mlm": False,
            "mlm_ratio": 1.0,
            "cmt_depth": 1,
            "id": False,
            "id_ratio": 0.0,
        }),
        "model": EasyDict({
            "embed_dim": 768,
            "use_gather": False,
            "softlabel_ratio": 0.5,
            "vocab_size": vocab_size,
            "tokenizer_type": "altclip",
            "num_classes": 1,
        }),
    })


def build_dummy_batch(processor, image_path: Path, captions, device: str):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=[image, image], return_tensors="pt")
    batch = {
        "image": inputs.pixel_values.to(device),
        "aug1": inputs.pixel_values.to(device),
        "caption": captions,
        "caption_bt": captions,
        "id": torch.arange(len(captions), device=device),
    }
    return batch


def embedding_sanity_check(model, tokenizer, sentences, device: str):
    tokenized = tokenizer(sentences, context_length=model.config.experiment.text_length)
    tokenized = tokenized.to(device)
    with torch.no_grad():
        embeds = model.encode_text(tokenized)
        embeds = F.normalize(embeds, dim=-1)
    similarity = embeds @ embeds.T
    return similarity.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--altclip_pretrained", type=str, default="BAAI/AltCLIP",
                        help="Path or HF repo id of AltCLIP (e.g., ./hf_cache/BAAI/AltCLIP)")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu device, e.g. cuda:0")
    parser.add_argument("--image", type=str, default=str(Path("image/intro.png")), help="Test image path")
    parser.add_argument("--text", nargs="*", default=["a person wearing a red jacket", "一个穿红色夹克的人"],
                        help="Texts to encode; default includes EN and ZH examples")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base_tokenizer = AutoTokenizer.from_pretrained(args.altclip_pretrained)
    config = build_config(device, vocab_size=base_tokenizer.vocab_size)
    clip_model, tokenizer, processor = build_altclip_clip(
        config, model_name=args.altclip_pretrained, device=device
    )

    captions = args.text
    batch = build_dummy_batch(processor, Path(args.image), captions, device)

    clip_model.eval()
    with torch.no_grad():
        outputs = clip_model(batch, alpha=0.5)
    print("Forward pass losses:", {k: float(v) for k, v in outputs.items()})

    similarity = embedding_sanity_check(clip_model, tokenizer, captions, device)
    print("Text embedding cosine similarity matrix:\n", similarity)
if __name__ == "__main__":
    main()
