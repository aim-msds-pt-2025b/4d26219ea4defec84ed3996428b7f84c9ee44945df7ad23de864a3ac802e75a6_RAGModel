"""
TF-IDF aware drift generation.

Creates realistic drift by modifying text to push TF-IDF features:
- Domain swap: inject domain-specific tokens
- OOV injection: tokens not in training vocab
- Style changes: affects tokenization & TF
- Topic proportion shift: change label mix
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import re
from typing import List, Iterable

# Domain-specific vocabulary pools
SPORTS = [
    "match",
    "league",
    "coach",
    "midfielder",
    "playoff",
    "goal",
    "assist",
    "transfer",
    "fixture",
    "derby",
]
BUSINESS = [
    "merger",
    "acquisition",
    "dividend",
    "earnings",
    "IPO",
    "guidance",
    "revenue",
    "EBITDA",
    "valuation",
    "buyback",
]
SCI = [
    "quantum",
    "genome",
    "neuron",
    "algorithm",
    "dataset",
    "nanotube",
    "plasma",
    "fusion",
    "catalyst",
    "entropy",
]
WORLD = [
    "summit",
    "treaty",
    "embassy",
    "sanctions",
    "referendum",
    "minister",
    "coalition",
    "border",
    "sovereignty",
    "bloc",
]

# OOV-like artifacts that wouldn't be in training vocab
URLS = ["https://t.co/xyz", "http://bit.ly/abc", "https://news.site/article"]
HASHTAGS = ["#breaking", "#analysis", "#trending", "#update"]
EMOJI = ["🙂", "🔥", "📈", "⚽", "🧪"]

DOMAIN_POOLS = {0: WORLD, 1: SPORTS, 2: BUSINESS, 3: SCI}


def inject_terms(
    text: str, terms: List[str], p_terms: float, rng: np.random.Generator
) -> str:
    """Inject domain-specific terms into text."""
    if not text or not isinstance(text, str):
        return text
    if rng.random() < p_terms:
        add = rng.choice(terms, size=rng.integers(1, 3), replace=True).tolist()
        return text + " " + " ".join(add)
    return text


def inject_oov(text: str, p_oov: float, rng: np.random.Generator) -> str:
    """Inject out-of-vocabulary tokens like URLs, hashtags, emoji."""
    if not text or not isinstance(text, str):
        return text
    bits: List[str] = []
    if rng.random() < p_oov:
        bits.append(rng.choice(URLS))
    if rng.random() < p_oov:
        bits.append(rng.choice(HASHTAGS))
    if rng.random() < p_oov / 2:
        bits.append(rng.choice(EMOJI))
    return text + (" " + " ".join(bits) if bits else "")


def tweak_style(
    text: str, rng: np.random.Generator, p_case: float = 0.25, p_punct: float = 0.25
) -> str:
    """Modify text style to affect tokenization patterns."""
    if not text or not isinstance(text, str):
        return text
    out = text
    if rng.random() < p_case:
        out = out.upper() if rng.random() < 0.5 else out.lower()
    if rng.random() < p_punct:
        out = re.sub(r"([.?!])", r"\1\1", out)  # duplicate sentence enders
    return out


def drift_text_tfidf_aware(
    series: pd.Series,
    labels: Iterable[int] | None,
    rng: np.random.Generator,
    p_domain: float = 0.6,  # inject domain tokens in 60% of rows
    p_oov: float = 0.4,  # inject OOV-like artifacts in 40% of rows
) -> pd.Series:
    """
    Create TF-IDF aware drift by modifying text content.

    Args:
        series: Text series to modify
        labels: Optional labels to guide domain injection
        rng: Random number generator
        p_domain: Probability of injecting domain-specific terms
        p_oov: Probability of injecting OOV tokens

    Returns:
        Modified text series with realistic drift
    """
    labels = list(labels) if labels is not None else [None] * len(series)
    out = []

    for s, y in zip(series.tolist(), labels):
        t = s
        # Domain shift: bias towards specific domains irrespective of original label
        pool = (
            DOMAIN_POOLS.get(int(y), SPORTS + BUSINESS)
            if y is not None
            else SPORTS + BUSINESS
        )
        t = inject_terms(t, pool, p_domain, rng)

        # OOV / social artifacts
        t = inject_oov(t, p_oov, rng)

        # Style noise that impacts tokenization mildly
        t = tweak_style(t, rng)

        out.append(t)

    return pd.Series(out, index=series.index)


def flip_labels(
    labels: pd.Series, rng: np.random.Generator, low: float = 0.05, high: float = 0.15
) -> pd.Series:
    """
    Create modest label drift by flipping a small percentage of labels.

    Args:
        labels: Label series to modify
        rng: Random number generator
        low: Minimum flip probability
        high: Maximum flip probability

    Returns:
        Modified labels with some flips
    """
    uniq = labels.unique().tolist()
    p = rng.uniform(low, high)
    mask = rng.random(len(labels)) < p
    out = labels.copy()

    for idx in labels[mask].index:
        cur = labels.loc[idx]
        others = [c for c in uniq if c != cur]
        if others:
            out.loc[idx] = rng.choice(others)

    return out
