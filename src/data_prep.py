#!/usr/bin/env python3
"""
src/data_prep.py
Foundations of Generative AI (ICT4403) - Assignment 3
Prepares and preprocesses the English-French parallel dataset (OPUS-100 en-fr).
Splits into Train, Validation, and Test sets and saves summary statistics.
"""

import os
import re
import json
import unicodedata
from collections import Counter
from datasets import load_dataset
import pandas as pd


def normalize_text(text: str) -> str:
    """Normalize unicode, strip excess whitespaces, and sanitize punctuation."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    # Remove zero-width spaces or control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_pair(en: str, fr: str, min_words: int = 3, max_words: int = 35) -> bool:
    """Filter out noise, excessively short/long sentences, and trivial copies."""
    if not en or not fr:
        return False
    # Avoid identical copies (source == target)
    if en.strip().lower() == fr.strip().lower():
        return False
    en_words = en.split()
    fr_words = fr.split()
    if not (min_words <= len(en_words) <= max_words):
        return False
    if not (min_words <= len(fr_words) <= max_words):
        return False
    # Length ratio filter (filter out mistranslations where length differs drastically)
    ratio = len(en_words) / max(len(fr_words), 1)
    if ratio < 0.35 or ratio > 2.8:
        return False
    return True


def prepare_dataset(
    output_dir: str = "data",
    train_size: int = 10000,
    val_size: int = 1000,
    test_size: int = 500,
    seed: int = 42,
):
    """Download, preprocess, filter and split the OPUS-100 en-fr dataset."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("Loading OPUS-100 (en-fr) dataset from Hugging Face...")
    print("=" * 60)

    try:
        raw_dataset = load_dataset("Helsinki-NLP/opus-100", "en-fr")
    except Exception as e:
        print(f"Error loading full opus-100: {e}. Trying train split stream...")
        raw_dataset = load_dataset("Helsinki-NLP/opus-100", "en-fr", split="train[:50000]")

    # Extract raw pairs
    print("Extracting and filtering sentence pairs...")
    splits_data = {"train": [], "validation": [], "test": []}

    if hasattr(raw_dataset, "keys") and "train" in raw_dataset:
        for split_name in ["train", "validation", "test"]:
            if split_name in raw_dataset:
                for item in raw_dataset[split_name]:
                    en = normalize_text(item["translation"].get("en", ""))
                    fr = normalize_text(item["translation"].get("fr", ""))
                    if is_valid_pair(en, fr):
                        splits_data[split_name].append({"en": en, "fr": fr})
    else:
        # Single split fallback
        for item in raw_dataset:
            en = normalize_text(item["translation"].get("en", ""))
            fr = normalize_text(item["translation"].get("fr", ""))
            if is_valid_pair(en, fr):
                splits_data["train"].append({"en": en, "fr": fr})

    # Deduplicate and sample deterministic subsets
    train_df = pd.DataFrame(splits_data["train"]).drop_duplicates(subset=["en", "fr"])
    val_df = pd.DataFrame(splits_data["validation"]).drop_duplicates(subset=["en", "fr"])
    test_df = pd.DataFrame(splits_data["test"]).drop_duplicates(subset=["en", "fr"])

    # If val/test are smaller or missing from slice, sample from train
    if len(val_df) < val_size or len(test_df) < test_size:
        print("Re-sampling validation and test from overall pool to ensure balance...")
        combined = pd.concat([train_df, val_df, test_df]).drop_duplicates(subset=["en", "fr"])
        combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        test_df = combined.iloc[:test_size]
        val_df = combined.iloc[test_size : test_size + val_size]
        train_df = combined.iloc[test_size + val_size : test_size + val_size + train_size]
    else:
        train_df = train_df.sample(n=min(train_size, len(train_df)), random_state=seed)
        val_df = val_df.sample(n=min(val_size, len(val_df)), random_state=seed)
        test_df = test_df.sample(n=min(test_size, len(test_df)), random_state=seed)

    print(f"Final dataset split counts:")
    print(f"  • Training pairs:   {len(train_df):,}")
    print(f"  • Validation pairs: {len(val_df):,}")
    print(f"  • Test pairs:       {len(test_df):,}")

    # Save to JSON Lines
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")

    train_df.to_json(train_path, orient="records", lines=True, force_ascii=False)
    val_df.to_json(val_path, orient="records", lines=True, force_ascii=False)
    test_df.to_json(test_path, orient="records", lines=True, force_ascii=False)

    # Compute descriptive statistics
    en_words_lens = train_df["en"].apply(lambda s: len(s.split())).tolist()
    fr_words_lens = train_df["fr"].apply(lambda s: len(s.split())).tolist()

    en_vocab = Counter([w.lower() for s in train_df["en"] for w in s.split()])
    fr_vocab = Counter([w.lower() for s in train_df["fr"] for w in s.split()])

    stats = {
        "dataset_name": "OPUS-100 (Helsinki-NLP/opus-100)",
        "source_language": "English (en)",
        "target_language": "French (fr)",
        "source_url": "https://huggingface.co/datasets/Helsinki-NLP/opus-100",
        "official_paper_ref": "Aulamo, M., Sulubacak, U., Virpioja, J., & Tiedemann, J. (2020). OPUS-100 corpus.",
        "splits": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
        },
        "english_stats": {
            "avg_word_length": float(sum(en_words_lens) / len(en_words_lens)),
            "min_word_length": int(min(en_words_lens)),
            "max_word_length": int(max(en_words_lens)),
            "unique_tokens": len(en_vocab),
        },
        "french_stats": {
            "avg_word_length": float(sum(fr_words_lens) / len(fr_words_lens)),
            "min_word_length": int(min(fr_words_lens)),
            "max_word_length": int(max(fr_words_lens)),
            "unique_tokens": len(fr_vocab),
        },
        "sample_pairs": train_df.head(3).to_dict(orient="records"),
    }

    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Saved dataset files and stats to {output_dir}/")
    print(f"Stats summary: En Avg Len={stats['english_stats']['avg_word_length']:.2f} words, "
          f"Fr Avg Len={stats['french_stats']['avg_word_length']:.2f} words.")
    return train_path, val_path, test_path, stats_path


if __name__ == "__main__":
    prepare_dataset()
