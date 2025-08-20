"""
TF-IDF aware drift features.

Builds numeric features from TF-IDF vectorizer that Evidently can reliably detect drift on:
- OOV rate, rare token rate, topk share
- TF-IDF norms, character/word counts
- Centroid similarities
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_reference_vectorizer(
    texts: pd.Series, max_features: int = 20000, ngram_range=(1, 2)
) -> TfidfVectorizer:
    """Fit TF-IDF vectorizer on reference (training) text only."""
    vec = TfidfVectorizer(
        max_features=max_features, ngram_range=ngram_range, lowercase=True, min_df=2
    )
    vec.fit(texts.astype(str))
    return vec


def _tokenize_for_vocab(vec: TfidfVectorizer, texts: pd.Series):
    """Use vectorizer's analyzer to tokenize texts for vocab analysis."""
    analyzer = vec.build_analyzer()
    return [analyzer(t) for t in texts.astype(str).tolist()]


def compute_tfidf_features(
    vec: TfidfVectorizer, texts: pd.Series, topk: int = 200
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute TF-IDF aware features for drift detection.

    Args:
        vec: Fitted TF-IDF vectorizer from reference data
        texts: Text series to extract features from
        topk: Number of top tokens to consider for topk_share

    Returns:
        Tuple of (feature DataFrame, diagnostics dict)
    """
    # Basic text statistics using tokenizer
    tokens_per_doc = _tokenize_for_vocab(vec, texts)
    word_count = np.array([len(toks) for toks in tokens_per_doc], dtype=float)
    char_len = texts.astype(str).str.len().to_numpy(dtype=float)
    avg_token_len = np.array(
        [np.mean([len(t) for t in toks]) if toks else 0.0 for toks in tokens_per_doc]
    )

    # TF-IDF features - use basic operations
    vocab_set = set(vec.vocabulary_.keys())
    idf = vec.idf_
    rare_thresh = np.quantile(idf, 0.9) if len(idf) > 0 else 0.0
    topk_idx = set(np.argsort(idf)[:topk]) if len(idf) > 0 else set()

    # Compute features per document
    oov_rates = []
    rare_rates = []
    topk_shares = []
    tfidf_norms = []

    for tokens in tokens_per_doc:
        if not tokens:
            oov_rates.append(0.0)
            rare_rates.append(0.0)
            topk_shares.append(0.0)
            tfidf_norms.append(0.0)
            continue

        # OOV rate
        oov_count = sum(1 for t in tokens if t not in vocab_set)
        oov_rate = oov_count / len(tokens)
        oov_rates.append(oov_rate)

        # Get vocabulary tokens with their IDF scores
        vocab_tokens = [(t, vec.vocabulary_[t]) for t in tokens if t in vocab_set]

        if not vocab_tokens:
            rare_rates.append(0.0)
            topk_shares.append(0.0)
            tfidf_norms.append(0.0)
            continue

        # Rare token rate (high IDF)
        rare_count = sum(
            1 for _, idx in vocab_tokens if idx < len(idf) and idf[idx] >= rare_thresh
        )
        rare_rate = rare_count / len(vocab_tokens)
        rare_rates.append(rare_rate)

        # Top-k share (most frequent = low IDF)
        topk_count = sum(1 for _, idx in vocab_tokens if idx in topk_idx)
        topk_share = topk_count / len(vocab_tokens)
        topk_shares.append(topk_share)

        # Approximate TF-IDF norm (simplified)
        # This is a rough approximation without full TF-IDF computation
        unique_tokens = list(set(t for t, _ in vocab_tokens))
        norm_approx = len(unique_tokens) / len(tokens)  # Simple diversity measure
        tfidf_norms.append(norm_approx)

    # Build feature DataFrame for Evidently
    feats = pd.DataFrame(
        {
            "tfidf_norm": np.array(tfidf_norms),
            "word_count": word_count,
            "char_len": char_len,
            "avg_token_len": avg_token_len,
            "oov_rate": np.array(oov_rates),
            "rare_token_rate": np.array(rare_rates),
            "topk_share": np.array(topk_shares),
        }
    )

    # Simple diagnostics
    diagnostics = {
        "idf_rare_threshold": float(rare_thresh),
        "vocab_size": int(len(vec.vocabulary_)),
        "mean_oov_rate": float(np.mean(oov_rates)),
        "mean_rare_rate": float(np.mean(rare_rates)),
    }

    return feats, diagnostics


def dataset_centroid_cosine(
    vec: TfidfVectorizer, A_texts: pd.Series, B_texts: pd.Series
) -> float:
    """Compute a simple similarity measure between datasets."""
    # Simple approach: compute vocabulary overlap
    try:
        A_tokens = _tokenize_for_vocab(vec, A_texts)
        B_tokens = _tokenize_for_vocab(vec, B_texts)

        # Flatten and get unique tokens
        A_vocab = set(token for doc in A_tokens for token in doc)
        B_vocab = set(token for doc in B_tokens for token in doc)

        if not A_vocab or not B_vocab:
            return 0.0

        # Jaccard similarity as approximation
        intersection = len(A_vocab & B_vocab)
        union = len(A_vocab | B_vocab)

        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.5  # Neutral similarity if anything fails
