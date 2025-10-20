"""
Small subset test with real Chinese translations from Zhipu AI.

This script uses actual Chinese translations to test multilingual retrieval.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from easydict import EasyDict
from transformers import AutoTokenizer

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_AUTO_CONVERSION", "1")

from model.altclip_adapter import build_altclip_clip


def load_bilingual_dataset(en_anno_file: str,
                           zh_anno_file: str,
                           image_root: str,
                           num_images: int = None,
                           seed: int = 42) -> Dict:
    """
    Load bilingual dataset (English + Chinese).

    Args:
        en_anno_file: Path to English annotation file
        zh_anno_file: Path to Chinese annotation file
        image_root: Path to image root
        num_images: Number of images to sample (None for all)
        seed: Random seed

    Returns:
        Dictionary with English and Chinese captions
    """
    print(f"\nLoading annotations...")
    print(f"  English: {en_anno_file}")
    print(f"  Chinese: {zh_anno_file}")

    # Load English
    with open(en_anno_file, 'r', encoding='utf-8') as f:
        en_data = json.load(f)

    # Load Chinese
    with open(zh_anno_file, 'r', encoding='utf-8') as f:
        zh_data = json.load(f)

    print(f"Total samples: {len(en_data)}")

    # Sample if needed
    if num_images is not None:
        random.seed(seed)
        indices = random.sample(range(len(en_data)), min(num_images, len(en_data)))
        en_data = [en_data[i] for i in indices]
        zh_data = [zh_data[i] for i in indices]
        print(f"Sampled {len(en_data)} images")

    # Build dataset
    images = []
    en_captions = []
    zh_captions = []
    img2person = []
    txt2person = []

    image_root = Path(image_root)

    for en_item, zh_item in zip(en_data, zh_data):
        assert en_item['image'] == zh_item['image'], "Data mismatch!"
        assert en_item['image_id'] == zh_item['image_id'], "Data mismatch!"

        image_path = image_root / en_item['image']
        person_id = en_item['image_id']

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        images.append(str(image_path))
        img2person.append(person_id)

        # Add captions
        for en_cap, zh_cap in zip(en_item['caption'], zh_item['caption']):
            en_captions.append(en_cap)
            zh_captions.append(zh_cap)
            txt2person.append(person_id)

    print(f"\nDataset loaded:")
    print(f"  Images: {len(images)}")
    print(f"  English captions: {len(en_captions)}")
    print(f"  Chinese captions: {len(zh_captions)}")
    print(f"  Unique persons: {len(set(img2person))}")

    return {
        'images': images,
        'en_captions': en_captions,
        'zh_captions': zh_captions,
        'img2person': torch.tensor(img2person, dtype=torch.long),
        'txt2person': torch.tensor(txt2person, dtype=torch.long),
    }


class BilingualTester:
    """Tester for bilingual retrieval."""

    def __init__(self, device: str = None):
        """Initialize tester."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load AltCLIP
        print("\nLoading AltCLIP model...")
        base_tokenizer = AutoTokenizer.from_pretrained("BAAI/AltCLIP")
        config = self._build_config(base_tokenizer.vocab_size)
        self.model, self.tokenizer, self.processor = build_altclip_clip(
            config, device=self.device
        )
        self.model.eval()
        print("✓ AltCLIP model loaded successfully")

    def _build_config(self, vocab_size: int) -> EasyDict:
        """Build config."""
        return EasyDict({
            "device": self.device,
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

    @torch.no_grad()
    def encode_images(self, image_paths: List[str]) -> torch.Tensor:
        """Encode images."""
        features = []
        for path in tqdm(image_paths, desc="Encoding images"):
            try:
                image = Image.open(path).convert('RGB')
                inputs = self.processor(images=image, return_tensors="pt")
                image_tensor = inputs.pixel_values.to(self.device)
                feat = self.model.encode_image(image_tensor)
                feat = F.normalize(feat, dim=-1)
                features.append(feat.cpu())
            except Exception as e:
                print(f"Error: {e}")
                features.append(torch.zeros(1, 768))
        return torch.cat(features, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts."""
        features = []
        batch_size = 32
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch = texts[i:i + batch_size]
            tokenized = self.tokenizer(batch, context_length=77)
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            feat = self.model.encode_text(tokenized)
            feat = F.normalize(feat, dim=-1)
            features.append(feat.cpu())
        return torch.cat(features, dim=0)

    def compute_metrics(self, text_feats, image_feats, txt2person, img2person):
        """Compute metrics."""
        device = self.device
        text_feats = text_feats.to(device)
        image_feats = image_feats.to(device)
        txt2person = txt2person.to(device)
        img2person = img2person.to(device)

        sims = text_feats @ image_feats.T
        index = torch.argsort(sims, dim=-1, descending=True)
        pred = img2person[index]
        matches = (txt2person.view(-1, 1).eq(pred)).long()

        def recall_k(m, k=1):
            mk = m[:, :k].sum(dim=-1)
            return 100.0 * torch.sum(mk > 0) / m.size(0)

        r1 = recall_k(matches, 1).item()
        r5 = recall_k(matches, 5).item()
        r10 = recall_k(matches, 10).item()

        # mAP
        real_num = matches.sum(dim=-1)
        cmc = matches.cumsum(dim=-1).float()
        order = torch.arange(1, matches.size(1) + 1, dtype=torch.long).to(device)
        cmc /= order
        cmc *= matches
        ap = cmc.sum(dim=-1) / real_num
        mAP = ap.mean() * 100.0

        return {
            'R@1': r1,
            'R@5': r5,
            'R@10': r10,
            'mAP': mAP.item()
        }

    def test_bilingual(self, dataset: Dict):
        """Test bilingual retrieval."""
        print("\n" + "="*80)
        print("BILINGUAL PERSON RETRIEVAL TEST")
        print("="*80)

        # Encode images once
        print("\n[1/4] Encoding images...")
        image_feats = self.encode_images(dataset['images'])

        # Encode English captions
        print("\n[2/4] Encoding English captions...")
        en_feats = self.encode_texts(dataset['en_captions'])

        # Encode Chinese captions
        print("\n[3/4] Encoding Chinese captions...")
        zh_feats = self.encode_texts(dataset['zh_captions'])

        # Compute metrics
        print("\n[4/4] Computing metrics...")

        en_metrics = self.compute_metrics(
            en_feats, image_feats,
            dataset['txt2person'], dataset['img2person']
        )

        zh_metrics = self.compute_metrics(
            zh_feats, image_feats,
            dataset['txt2person'], dataset['img2person']
        )

        # Print results
        self._print_results(en_metrics, zh_metrics)

        # Compare semantic similarity
        self._compare_semantics(
            en_feats, zh_feats,
            dataset['en_captions'][:10],
            dataset['zh_captions'][:10]
        )

        return en_metrics, zh_metrics

    def _print_results(self, en_metrics, zh_metrics):
        """Print comparison results."""
        print("\n" + "="*80)
        print("RETRIEVAL RESULTS")
        print("="*80)

        print(f"\nEnglish:")
        print(f"  R@1:  {en_metrics['R@1']:.2f}%")
        print(f"  R@5:  {en_metrics['R@5']:.2f}%")
        print(f"  R@10: {en_metrics['R@10']:.2f}%")
        print(f"  mAP:  {en_metrics['mAP']:.2f}%")

        print(f"\nChinese:")
        print(f"  R@1:  {zh_metrics['R@1']:.2f}%")
        print(f"  R@5:  {zh_metrics['R@5']:.2f}%")
        print(f"  R@10: {zh_metrics['R@10']:.2f}%")
        print(f"  mAP:  {zh_metrics['mAP']:.2f}%")

        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        print(f"\n{'Metric':<10} {'English':>12} {'Chinese':>12} {'Difference':>12}")
        print("-" * 50)

        for m in ['R@1', 'R@5', 'R@10', 'mAP']:
            en_v = en_metrics[m]
            zh_v = zh_metrics[m]
            diff = zh_v - en_v
            print(f"{m:<10} {en_v:>11.2f}% {zh_v:>11.2f}% {diff:>+11.2f}%")

        avg_diff = sum(zh_metrics[m] - en_metrics[m] for m in ['R@1', 'R@5', 'R@10', 'mAP']) / 4

        print("\n" + "="*80)
        print("Analysis:")
        if abs(avg_diff) < 2.0:
            print(f"✓ Excellent! Chinese and English show similar performance (diff < 2%)")
            print(f"  Average difference: {avg_diff:+.2f}%")
            print(f"  This confirms AltCLIP's strong multilingual capability!")
        elif avg_diff > 0:
            print(f"  Chinese performs {avg_diff:.2f}% better on average")
        else:
            print(f"  English performs {-avg_diff:.2f}% better on average")
        print("="*80)

    def _compare_semantics(self, en_feats, zh_feats, en_caps, zh_caps):
        """Compare semantic similarity."""
        print("\n" + "="*80)
        print("SEMANTIC SIMILARITY (Cross-lingual)")
        print("="*80)

        en_feats = en_feats.to(self.device)
        zh_feats = zh_feats.to(self.device)

        sims = (en_feats * zh_feats).sum(dim=-1).cpu()

        print("\nSample pairs:")
        for i, (en, zh, sim) in enumerate(zip(en_caps, zh_caps, sims), 1):
            print(f"\n[{i}] Similarity: {sim:.4f}")
            print(f"    EN: {en}")
            print(f"    ZH: {zh}")

        avg = sims.mean().item()
        print(f"\n{'='*80}")
        print(f"Average cross-lingual similarity: {avg:.4f}")

        if avg > 0.95:
            print("✓ Excellent! Nearly perfect semantic alignment")
        elif avg > 0.90:
            print("✓ Very good semantic alignment")
        elif avg > 0.85:
            print("✓ Good semantic alignment")
        else:
            print("⚠ Semantic alignment needs improvement")
        print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_anno", type=str,
                       default="/data1/dataset/cuhkpedes/cuhk_test.json")
    parser.add_argument("--zh_anno", type=str,
                       default="tests/translations/cuhk_test_zh_sample.json")
    parser.add_argument("--image_root", type=str,
                       default="/data1/dataset/cuhkpedes/imgs")
    parser.add_argument("--num_images", type=int, default=None,
                       help="Number of images (None for all in zh_anno)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load dataset
    dataset = load_bilingual_dataset(
        args.en_anno, args.zh_anno, args.image_root,
        args.num_images, args.seed
    )

    # Test
    tester = BilingualTester(device=args.device)
    en_metrics, zh_metrics = tester.test_bilingual(dataset)

    print("\n✓ Test completed!")
    print("\nNote: This is zero-shot evaluation (pretrained AltCLIP, not fine-tuned)")


if __name__ == "__main__":
    main()
