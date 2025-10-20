import pathlib
import sys

import torch
from easydict import EasyDict

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model import tbps_model
from model.tbps_model import clip_vitb


class _DummyEDA:
    def random_deletion(self, sentence, alpha):
        return sentence


tbps_model.EDA = _DummyEDA


def make_base_config():
    experiment = EasyDict({
        "input_resolution": [64, 64],
        "simclr_mlp": [512, 128, 512],
        "simclr_temperature": 0.1,
        "dropout": 0.0,
        "eda_alpha": 0.0,
        "back_trans": False,
        "backtrans_p": 0.0,
        "text_length": 16,
        "mixgen": False,
        "mixgen_type": "cat",
        "mixgen_p": 0.0,
        "mixgen_ratio": 0.0,
        "mvs_image": False,
        "nitc_ratio": 1.0,
        "ss": False,
        "ss_ratio": 0.0,
        "citc": False,
        "citc_lambda1": 0.25,
        "citc_lambda2": 0.25,
        "citc_ratio": 1.0,
        "ritc": False,
        "ritc_eps": 1.0e-2,
        "ritc_ratio": 1.0,
        "mlm": False,
        "mlm_ratio": 1.0,
        "cmt_depth": 1,
        "id": False,
        "id_ratio": 0.0,
    })

    model = EasyDict({
        "embed_dim": 128,
        "use_gather": False,
        "softlabel_ratio": 0.5,
        "vocab_size": 49408,
    })

    config = EasyDict({
        "device": "cpu",
        "misc": EasyDict({"seed": 1}),
        "experiment": experiment,
        "model": model,
    })

    return config


def make_batch(batch_size=2):
    captions = ["a person wearing a red coat", "a person wearing a blue coat"]
    images = torch.randn(batch_size, 3, 64, 64)
    batch = {
        "image": images,
        "aug1": images.clone(),
        "caption": captions,
        "caption_bt": captions,
        "id": torch.arange(batch_size, dtype=torch.long),
    }
    return batch


def test_forward_produces_nitc_loss():
    config = make_base_config()
    model = clip_vitb(config, num_classes=4)
    batch = make_batch()
    outputs = model(batch, alpha=0.5)
    assert "nitc_loss" in outputs
    assert outputs["nitc_loss"].shape == ()


def test_forward_with_citc_path():
    config = make_base_config()
    config.experiment.citc = True
    model = clip_vitb(config, num_classes=4)
    batch = make_batch()
    outputs = model(batch, alpha=0.5)
    assert "citc_loss" in outputs
    assert outputs["citc_loss"].shape == ()


def test_forward_with_ritc_path():
    config = make_base_config()
    config.experiment.ritc = True
    model = clip_vitb(config, num_classes=4)
    batch = make_batch()
    outputs = model(batch, alpha=0.5)
    assert "ritc_loss" in outputs
    assert outputs["ritc_loss"].shape == ()


def test_forward_with_mlm_path():
    config = make_base_config()
    config.experiment.mlm = True
    model = clip_vitb(config, num_classes=4)
    batch = make_batch()
    outputs = model(batch, alpha=0.5)
    assert "mlm_loss" in outputs
    assert outputs["mlm_loss"].shape == ()
