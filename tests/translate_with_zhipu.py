"""
Translate captions using Zhipu AI API (GLM-4-Flash).
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

try:
    from zhipuai import ZhipuAI
except ImportError:
    print("Error: zhipuai package not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "zhipuai"])
    from zhipuai import ZhipuAI


class ZhipuTranslator:
    """Translator using Zhipu AI API."""

    def __init__(self, api_key: str, model: str = "glm-4-flash"):
        """
        Initialize translator.

        Args:
            api_key: Zhipu AI API key
            model: Model name (default: glm-4-flash)
        """
        self.client = ZhipuAI(api_key=api_key)
        self.model = model
        print(f"✓ Initialized Zhipu AI translator with model: {model}")

    def translate_single(self, text: str, retry: int = 3) -> str:
        """
        Translate a single text.

        Args:
            text: English text to translate
            retry: Number of retries on failure

        Returns:
            Translated Chinese text
        """
        prompt = f"""请将以下英文行人描述翻译成中文。要求：
1. 保持描述的准确性和细节
2. 使用自然流畅的中文表达
3. 只输出翻译结果，不要添加任何解释或注释

英文描述：{text}

中文翻译："""

        for attempt in range(retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent translation
                )

                translation = response.choices[0].message.content.strip()

                # Remove any quotation marks that might be added
                translation = translation.strip('"').strip("'").strip()

                return translation

            except Exception as e:
                if attempt < retry - 1:
                    print(f"\nWarning: Translation failed (attempt {attempt + 1}/{retry}): {e}")
                    time.sleep(2)  # Wait before retry
                else:
                    print(f"\nError: Failed to translate after {retry} attempts: {text}")
                    return f"[翻译失败] {text}"

        return f"[翻译失败] {text}"

    def translate_batch(self, texts: List[str], max_workers: int = 20) -> List[str]:
        """
        Translate a batch of texts with parallel processing.

        Args:
            texts: List of English texts
            max_workers: Number of parallel workers (default: 20)

        Returns:
            List of translated Chinese texts
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        translations = [None] * len(texts)

        print(f"\nTranslating {len(texts)} captions with {max_workers} parallel workers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.translate_single, text): i
                for i, text in enumerate(texts)
            }

            # Collect results with progress bar
            with tqdm(total=len(texts), desc="Translating") as pbar:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        translation = future.result()
                        translations[index] = translation
                    except Exception as e:
                        print(f"\nError translating text {index}: {e}")
                        translations[index] = f"[翻译失败] {texts[index]}"
                    pbar.update(1)

        return translations


def translate_cuhk_pedes_subset(anno_file: str,
                                output_file: str,
                                api_key: str,
                                num_samples: int = None,
                                model: str = "glm-4-flash",
                                max_workers: int = 20):
    """
    Translate CUHK-PEDES annotations to Chinese.

    Args:
        anno_file: Path to original annotation file
        output_file: Path to output translated file
        api_key: Zhipu AI API key
        num_samples: Number of samples to translate (None for all)
        model: Model name
    """
    # Load annotations
    print(f"Loading annotations from {anno_file}...")
    with open(anno_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # Limit samples if specified
    if num_samples is not None and num_samples < len(data):
        data = data[:num_samples]
        print(f"Translating first {num_samples} samples...")

    # Initialize translator
    translator = ZhipuTranslator(api_key, model)

    # Collect all unique captions
    caption_to_idx = {}
    all_captions = []

    for item in data:
        for caption in item['caption']:
            if caption not in caption_to_idx:
                caption_to_idx[caption] = len(all_captions)
                all_captions.append(caption)

    print(f"Unique captions to translate: {len(all_captions)}")

    # Translate with parallel processing
    translations = translator.translate_batch(all_captions, max_workers=max_workers)

    # Build translation dictionary
    trans_dict = {cap: trans for cap, trans in zip(all_captions, translations)}

    # Apply translations to data
    translated_data = []
    for item in data:
        new_item = item.copy()
        new_item['caption'] = [trans_dict[cap] for cap in item['caption']]
        new_item['caption_en'] = item['caption']  # Keep original English
        translated_data.append(new_item)

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Translated annotations saved to: {output_path}")
    print(f"  Total samples: {len(translated_data)}")
    print(f"  Unique translations: {len(trans_dict)}")

    # Show some samples
    print(f"\nSample translations:")
    print("=" * 80)
    for i, (en, zh) in enumerate(list(trans_dict.items())[:5], 1):
        print(f"\n[{i}] EN: {en}")
        print(f"    ZH: {zh}")

    return translated_data, trans_dict


def translate_caption_list(captions: List[str],
                          api_key: str,
                          output_file: str = None,
                          model: str = "glm-4-flash") -> List[str]:
    """
    Translate a list of captions.

    Args:
        captions: List of English captions
        api_key: Zhipu AI API key
        output_file: Optional output JSON file
        model: Model name

    Returns:
        List of Chinese translations
    """
    translator = ZhipuTranslator(api_key, model)
    translations = translator.translate_batch(captions)

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        trans_dict = {en: zh for en, zh in zip(captions, translations)}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(trans_dict, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Translation dictionary saved to: {output_path}")

    return translations


def main():
    parser = argparse.ArgumentParser(description="Translate captions using Zhipu AI")
    parser.add_argument("--api_key", type=str,
                       required=True,
                       help="Zhipu AI API key (required)")
    parser.add_argument("--model", type=str, default="glm-4-flash",
                       help="Model name (default: glm-4-flash)")
    parser.add_argument("--anno_file", type=str,
                       default="/data1/dataset/cuhkpedes/cuhk_test.json",
                       help="Path to annotation file")
    parser.add_argument("--output_file", type=str,
                       default="tests/translations/cuhk_test_zh.json",
                       help="Path to output translated file")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to translate (None for all)")
    parser.add_argument("--max_workers", type=int, default=1000,
                       help="Number of parallel workers (default: 1000)")
    parser.add_argument("--test", action="store_true",
                       help="Test with a few samples first")

    args = parser.parse_args()

    # Test mode: translate just 5 samples
    if args.test:
        print("Running in TEST mode (5 samples)...")
        args.num_samples = 5

    # Translate
    translate_cuhk_pedes_subset(
        args.anno_file,
        args.output_file,
        args.api_key,
        args.num_samples,
        args.model,
        args.max_workers
    )


if __name__ == "__main__":
    main()
