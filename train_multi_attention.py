#!/usr/bin/env python3
"""
Train and compare multiple image captioning models that only differ by the attention block.
This script reuses precomputed CNN features (img_features.pkl) and a pre-fit tokenizer
(tokenizer.pkl) so you can train several attention variants without rebuilding dataset artifacts.

Example:
python train_multi_attention.py
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import re
from dataclasses import dataclass, asdict, field
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

import caption_core as core

SUPPORTED_ATTENTION_TYPES = ("luong", "bahdanau", "scaled_dot", "multihead", "concatenate")

DEFAULT_DATASET_DIR = Path("results")
DEFAULT_OUTPUT_DIR = Path("results/attention_runs")

@dataclass
class TrainingResult:
    attention: str
    epochs_run: int
    best_epoch: int
    best_val_loss: float
    best_train_loss: float
    final_val_loss: float
    final_train_loss: float
    model_path: str
    history_path: str
    log_path: str

    def to_row(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class TrainingConfig:
    captions_path: Path
    features_path: Path
    tokenizer_path: Path
    output_dir: Path = DEFAULT_OUTPUT_DIR
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    train_split: float = 0.9
    random_seed: int = 42
    attention_types: Sequence[str] = field(default_factory=lambda: list(SUPPORTED_ATTENTION_TYPES))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def safe_attention_name(kind: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", kind.lower())
    cleaned = cleaned.strip("_")
    return cleaned or "attention"


def load_image_features(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    with path.open("rb") as f:
        raw = pickle.load(f)
    features: Dict[str, np.ndarray] = {}
    for image_id, feat in raw.items():
        arr = np.asarray(feat, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim > 1:
            arr = arr.reshape(-1)
        features[image_id] = arr
    if not features:
        raise ValueError("No entries were found in the feature file.")
    return features


def load_tokenizer_file(path: Path) -> Tokenizer:
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")
    with path.open("rb") as f:
        tokenizer = pickle.load(f)
    if not getattr(tokenizer, "word_index", None):
        raise ValueError("Loaded tokenizer is missing a word_index. Ensure it was fitted before serialization.")
    return tokenizer


def clean_caption(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [word for word in text.split() if len(word) > 1]
    if not tokens:
        return ""
    return f"startseq {' '.join(tokens)} endseq"


def load_and_prepare_captions(captions_path: Path, feature_ids: Iterable[str]) -> Dict[str, List[str]]:
    if not captions_path.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_path}")
    feature_set = set(feature_ids)
    mapping: Dict[str, List[str]] = {}
    with captions_path.open("r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 3:
                continue
            image_id = parts[0].split(".")[0].strip()
            if image_id not in feature_set:
                continue
            caption = clean_caption(parts[2])
            if not caption:
                continue
            mapping.setdefault(image_id, []).append(caption)
    if not mapping:
        raise ValueError("No captions matched the available image features.")
    return mapping


def tokenize_with_existing_tokenizer(
    caption_map: Dict[str, List[str]], tokenizer: Tokenizer
) -> Tuple[Dict[str, List[List[int]]], int, int]:
    tokenized: Dict[str, List[List[int]]] = {}
    lengths: List[int] = []
    for image_id, captions in caption_map.items():
        seqs = tokenizer.texts_to_sequences(captions)
        seqs = [seq for seq in seqs if len(seq) > 1]
        if seqs:
            tokenized[image_id] = seqs
            lengths.extend(len(seq) for seq in seqs)
    if not tokenized or not lengths:
        raise ValueError(
            "Tokenizer did not yield any valid caption sequences. "
            "Ensure the tokenizer was fit on the same caption vocabulary."
        )
    max_len = max(lengths)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenized, vocab_size, max_len


def split_ids(image_ids: Sequence[str], train_split: float, seed: int) -> Tuple[List[str], List[str]]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be within (0, 1).")
    ids = list(image_ids)
    if len(ids) < 2:
        raise ValueError("At least two images are required to perform a split.")
    rng = random.Random(seed)
    rng.shuffle(ids)
    split_idx = max(1, int(len(ids) * train_split))
    split_idx = min(split_idx, len(ids) - 1)
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]
    return train_ids, val_ids


def count_pairs(keys: Sequence[str], tokenized_captions: Dict[str, List[List[int]]]) -> int:
    total = 0
    for image_id in keys:
        for seq in tokenized_captions[image_id]:
            if len(seq) > 1:
                total += len(seq) - 1
    return total


def data_generator_onehot(
    data_keys: Sequence[str],
    tokenized_captions: Dict[str, List[List[int]]],
    features: Dict[str, np.ndarray],
    max_caption_length: int,
    vocab_size: int,
    batch_size: int,
):
    X1_batch, X2_batch, y_batch = [], [], []
    keys = list(data_keys)
    while True:
        random.shuffle(keys)
        for image_id in keys:
            feat_vec = np.asarray(features[image_id], dtype=np.float32)
            for seq in tokenized_captions[image_id]:
                if len(seq) < 2:
                    continue
                for i in range(1, len(seq)):
                    in_seq = pad_sequences([seq[:i]], maxlen=max_caption_length, padding="post")[0]
                    out_tok = seq[i]
                    X1_batch.append(feat_vec)
                    X2_batch.append(in_seq.astype("int32", copy=False))
                    y_batch.append(to_categorical(out_tok, num_classes=vocab_size))
                    if len(X1_batch) == batch_size:
                        yield (
                            (
                                np.asarray(X1_batch, dtype=np.float32),
                                np.asarray(X2_batch, dtype=np.int32),
                            ),
                            np.asarray(y_batch, dtype=np.float32),
                        )
                        X1_batch, X2_batch, y_batch = [], [], []
        if X1_batch:
            yield (
                (
                    np.asarray(X1_batch, dtype=np.float32),
                    np.asarray(X2_batch, dtype=np.int32),
                ),
                np.asarray(y_batch, dtype=np.float32),
            )
            X1_batch, X2_batch, y_batch = [], [], []


def validate_attention_list(attention_types: Sequence[str] | None) -> List[str]:
    if attention_types is None:
        normalized = list(SUPPORTED_ATTENTION_TYPES)
    else:
        normalized = [att.strip().lower() for att in attention_types if att.strip()]
    if not normalized:
        normalized = list(SUPPORTED_ATTENTION_TYPES)
    invalid = sorted(set(normalized) - set(SUPPORTED_ATTENTION_TYPES))
    if invalid:
        raise ValueError(
            f"Unsupported attention types: {', '.join(invalid)}. "
            f"Supported values: {', '.join(SUPPORTED_ATTENTION_TYPES)}"
        )
    return normalized


def summarize_history(
    attention: str,
    history: tf.keras.callbacks.History,
    model_path: Path,
    history_path: Path,
    log_path: Path,
) -> TrainingResult:
    train_losses = history.history.get("loss", [])
    val_losses = history.history.get("val_loss", [])
    epochs_run = len(train_losses)
    if val_losses:
        best_idx = int(np.argmin(val_losses))
        best_val = float(val_losses[best_idx])
    else:
        best_idx = epochs_run - 1
        best_val = float("nan")
    best_train = float(train_losses[best_idx]) if train_losses else float("nan")
    final_val = float(val_losses[-1]) if val_losses else float("nan")
    final_train = float(train_losses[-1]) if train_losses else float("nan")
    return TrainingResult(
        attention=attention,
        epochs_run=epochs_run,
        best_epoch=best_idx + 1,
        best_val_loss=best_val,
        best_train_loss=best_train,
        final_val_loss=final_val,
        final_train_loss=final_train,
        model_path=str(model_path.resolve()),
        history_path=str(history_path.resolve()),
        log_path=str(log_path.resolve()),
    )


def write_comparison_artifacts(results: List[TrainingResult], output_dir: Path) -> None:
    if not results:
        return
    rows = [res.to_row() for res in results]
    comparison_json = output_dir / "attention_comparison.json"
    comparison_csv = output_dir / "attention_comparison.csv"
    comparison_json.write_text(json.dumps(rows, indent=2))
    with comparison_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_comparison_table(results: List[TrainingResult]) -> None:
    if not results:
        print("No models were trained.")
        return
    header = f"{'Attention':<15}{'Best Ep':<8}{'Best Val':<12}{'Final Val':<12}{'Model'}"
    print("\nModel comparison")
    print(header)
    print("-" * len(header))
    for res in results:
        print(
            f"{res.attention:<15}"
            f"{res.best_epoch:<8}"
            f"{res.best_val_loss:<12.4f}"
            f"{res.final_val_loss:<12.4f}"
            f"{res.model_path}"
        )


def train_attention_variants(config: TrainingConfig) -> List[TrainingResult]:
    set_global_seed(config.random_seed)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading precomputed CNN features...")
    features = load_image_features(config.features_path)
    print(f"Loaded features for {len(features)} images.")

    print("Loading and cleaning captions...")
    caption_map = load_and_prepare_captions(config.captions_path, features.keys())
    total_caps = sum(len(caps) for caps in caption_map.values())
    print(f"Loaded {total_caps} cleaned captions.")

    tokenizer = load_tokenizer_file(config.tokenizer_path)
    print(f"Loaded tokenizer from {config.tokenizer_path}")
    tokenized_captions, vocab_size, max_caption_length = tokenize_with_existing_tokenizer(caption_map, tokenizer)

    # Ensure we only keep images that have both features and tokenized captions
    caption_ids = set(tokenized_captions.keys())
    feature_ids = set(features.keys())
    shared_ids = sorted(caption_ids & feature_ids)
    if not shared_ids:
        raise ValueError("No overlap between tokenizer-ready captions and available image features.")
    missing_features = caption_ids - feature_ids
    missing_captions = feature_ids - caption_ids
    if missing_features:
        print(f"Warning: {len(missing_features)} caption entries missing features. They will be skipped.")
    if missing_captions:
        print(f"Warning: {len(missing_captions)} feature vectors without captions. They will be skipped.")
    tokenized_captions = {img_id: tokenized_captions[img_id] for img_id in shared_ids}
    features = {img_id: features[img_id] for img_id in shared_ids}

    image_ids = list(tokenized_captions.keys())
    train_ids, val_ids = split_ids(image_ids, config.train_split, config.random_seed)
    print(f"Train images: {len(train_ids)} | Val images: {len(val_ids)}")

    train_pairs = count_pairs(train_ids, tokenized_captions)
    val_pairs = count_pairs(val_ids, tokenized_captions)
    if train_pairs == 0 or val_pairs == 0:
        raise ValueError("Insufficient token pairs for training/validation. Check your data preparation.")
    steps_per_epoch = max(1, ceil(train_pairs / config.batch_size))
    validation_steps = max(1, ceil(val_pairs / config.batch_size))
    print(f"Steps/epoch: {steps_per_epoch} | Val steps: {validation_steps}")

    results: List[TrainingResult] = []
    trained_models_metadata: List[Dict[str, str]] = []

    attention_types = validate_attention_list(config.attention_types)

    for attention in attention_types:
        tf.keras.backend.clear_session()
        print(f"\n=== Training attention variant: {attention} ===")
        model = core.build_image_caption_model(
            vocab_size=vocab_size,
            max_caption_length=max_caption_length,
            attention_kind=attention,
            learning_rate=config.learning_rate,
        )

        train_gen = data_generator_onehot(
            train_ids, tokenized_captions, features, max_caption_length, vocab_size, config.batch_size
        )
        val_gen = data_generator_onehot(
            val_ids, tokenized_captions, features, max_caption_length, vocab_size, config.batch_size
        )

        safe_name = safe_attention_name(attention)
        csv_log_path = output_dir / f"{safe_name}_training_log.csv"
        history_path = output_dir / f"{safe_name}_history.json"
        model_path = output_dir / f"caption_{safe_name}.keras"

        callbacks = [
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
            CSVLogger(str(csv_log_path), append=False),
        ]

        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            epochs=config.epochs,
            verbose=1,
            callbacks=callbacks,
        )

        model.save(model_path)
        history_path.write_text(json.dumps(history.history, indent=2))

        results.append(summarize_history(attention, history, model_path, history_path, csv_log_path))
        trained_models_metadata.append({"attention": attention, "model_path": str(model_path.resolve())})

        print(f"Saved model to {model_path}")

    if trained_models_metadata:
        print("\nTrained model artifacts:")
        for item in trained_models_metadata:
            print(f" - {item['attention']}: {item['model_path']}")

    return results


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multiple attention-based caption models on the same dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing captions.csv, img_features.pkl, tokenizer.pkl (default: results/).",
    )
    parser.add_argument(
        "--captions-path",
        type=Path,
        help="Optional explicit path to captions.csv (overrides --dataset-dir).",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        help="Optional explicit path to img_features.pkl (overrides --dataset-dir).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="Optional explicit path to tokenizer.pkl (overrides --dataset-dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to save trained models, logs, and histories (default: results/attention_runs).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size (default: 64).")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per attention variant (default: 20).")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Fraction of images used for training (default: 0.9).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for shuffling/splitting (default: 42).",
    )
    parser.add_argument(
        "--attention-types",
        nargs="+",
        default=None,
        help="Subset of attention mechanisms to train (default: all).",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    captions_path = (
        Path(args.captions_path).expanduser().resolve() if args.captions_path else (dataset_dir / "captions.csv")
    )
    features_path = (
        Path(args.features_path).expanduser().resolve() if args.features_path else (dataset_dir / "img_features.pkl")
    )
    tokenizer_path = (
        Path(args.tokenizer_path).expanduser().resolve() if args.tokenizer_path else (dataset_dir / "tokenizer.pkl")
    )
    output_dir = Path(args.output_dir).expanduser().resolve()

    attention_types = args.attention_types
    if attention_types:
        if len(attention_types) == 1 and attention_types[0].lower() == "all":
            attention_types = list(SUPPORTED_ATTENTION_TYPES)
        else:
            attention_types = [att.lower() for att in attention_types]
    else:
        attention_types = list(SUPPORTED_ATTENTION_TYPES)

    return TrainingConfig(
        captions_path=captions_path,
        features_path=features_path,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_split=args.train_split,
        random_seed=args.random_seed,
        attention_types=attention_types,
    )


def main():
    args = parse_cli_args()
    config = config_from_args(args)

    print("=" * 80)
    print("Training Image Captioning Models with Multiple Attention Mechanisms")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Captions: {config.captions_path}")
    print(f"  Features: {config.features_path}")
    print(f"  Tokenizer: {config.tokenizer_path}")
    print(f"  Output Dir: {config.output_dir}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Train Split: {config.train_split}")
    print(f"  Random Seed: {config.random_seed}")
    print(f"  Attention Types: {', '.join(validate_attention_list(config.attention_types))}")
    print("=" * 80)

    results = train_attention_variants(config)
    print_comparison_table(results)
    write_comparison_artifacts(results, config.output_dir)
    print(f"\nAll results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
