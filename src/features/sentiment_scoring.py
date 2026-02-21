"""FinBERT sentiment scoring for news headlines.

Scores headlines using ProsusAI/finbert with GPU-optimized batching.
Handles OOM gracefully with adaptive batch sizing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_finbert(model_name: str = "ProsusAI/finbert", device: str = "auto"):
    """Load FinBERT model and tokenizer.

    Args:
        model_name: HuggingFace model identifier.
        device: Device string ("auto", "cuda", "cpu").

    Returns:
        Tuple of (model, tokenizer, resolved_device).
    """
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    logger.info(f"Loading {model_name} on {resolved_device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(resolved_device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {param_count / 1e6:.1f}M parameters on {resolved_device}")

    return model, tokenizer, resolved_device


def score_batch(
    headlines: list[str],
    model,
    tokenizer,
    device: str,
    max_length: int = 128,
) -> list[dict]:
    """Score a batch of headlines with FinBERT.

    Args:
        headlines: List of headline strings.
        model: Loaded FinBERT model.
        tokenizer: Loaded tokenizer.
        device: Device to run inference on.
        max_length: Maximum token length.

    Returns:
        List of dicts with keys [positive, negative, neutral, sentiment_score].
    """
    inputs = tokenizer(
        headlines,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

    # FinBERT classes: [positive, negative, neutral]
    results: list[dict] = []
    for i in range(len(headlines)):
        pos, neg, neu = float(probs[i, 0]), float(probs[i, 1]), float(probs[i, 2])
        results.append({
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "sentiment_score": pos - neg,
        })

    return results


def score_all_headlines(
    input_path: Path,
    output_path: Path,
    config: dict,
) -> None:
    """Score all headlines and save results.

    Processes headline Parquet files with adaptive batch sizing.
    For rows without extracted headlines, normalizes AvgTone as fallback.

    Args:
        input_path: Path to headlines directory or single Parquet file.
        output_path: Path to save scored Parquet files.
        config: Configuration dictionary.
    """
    sent_cfg = config["sentiment"]
    batch_size = sent_cfg.get("batch_size", 16)
    max_length = sent_cfg.get("max_length", 128)
    max_batch = batch_size * 2  # Upper limit for adaptive sizing

    # Discover input files
    input_path = Path(input_path)
    if input_path.is_file():
        parquet_files = [input_path]
    else:
        parquet_files = sorted(input_path.rglob("*.parquet"))

    if not parquet_files:
        logger.warning(f"No headline files found at {input_path}")
        return

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer, device = load_finbert(
        model_name=sent_cfg.get("model_name", "ProsusAI/finbert"),
        device=sent_cfg.get("device", "auto"),
    )

    total_scored = 0
    total_fallback = 0

    for pf in tqdm(parquet_files, desc="Scoring files"):
        df = pd.read_parquet(pf)
        if df.empty:
            continue

        # Initialize score columns
        df["positive"] = np.nan
        df["negative"] = np.nan
        df["neutral"] = np.nan
        df["sentiment_score"] = np.nan

        # Score rows that have extracted headlines
        has_hl = df["has_headline"].fillna(False)
        hl_indices = df.index[has_hl].tolist()

        if hl_indices:
            headlines = df.loc[hl_indices, "headline"].tolist()
            all_scores: list[dict] = []

            # Process in adaptive batches
            current_batch = batch_size
            i = 0
            while i < len(headlines):
                batch = headlines[i : i + current_batch]
                try:
                    scores = score_batch(batch, model, tokenizer, device, max_length)
                    all_scores.extend(scores)
                    i += current_batch
                    # Grow batch size on success (up to max)
                    current_batch = min(current_batch + current_batch // 4, max_batch)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    current_batch = max(1, current_batch // 2)
                    logger.warning(f"OOM: reduced batch size to {current_batch}")

            # Assign scores back
            for idx, score_dict in zip(hl_indices, all_scores):
                df.at[idx, "positive"] = score_dict["positive"]
                df.at[idx, "negative"] = score_dict["negative"]
                df.at[idx, "neutral"] = score_dict["neutral"]
                df.at[idx, "sentiment_score"] = score_dict["sentiment_score"]

            total_scored += len(hl_indices)

        # Fallback: normalize AvgTone to [-1, 1] for rows without headlines
        no_hl = ~has_hl
        if no_hl.any():
            # AvgTone typically ranges -10 to +10; clip and scale
            avg_tone = df.loc[no_hl, "avg_tone"].clip(-10, 10) / 10.0
            df.loc[no_hl, "sentiment_score"] = avg_tone
            total_fallback += no_hl.sum()

        # Save with same partition structure
        if input_path.is_file():
            out_file = output_path / pf.name.replace("headlines_", "sentiment_")
        else:
            rel_path = pf.relative_to(input_path)
            out_file = output_path / rel_path.parent / pf.name.replace("headlines_", "sentiment_")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_file, engine="pyarrow", compression="snappy", index=False)

    logger.info(
        f"Sentiment scoring complete: {total_scored} FinBERT scored, "
        f"{total_fallback} AvgTone fallback"
    )
