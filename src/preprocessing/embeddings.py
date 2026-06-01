"""Load pretrained vectors and align them with the experiment vocabulary."""

from pathlib import Path

import numpy as np
import torch

from src.utils.vocabulary import Vocabulary, PAD_TOKEN


def l2_normalize_matrix(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize each row independently while keeping zero rows stable."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def load_text_embeddings(path: str | Path, expected_dim: int | None = None) -> dict[str, np.ndarray]:
    """
    Legge embedding testuali tipo .vec.

    Supporta sia file con header:
        vocab_size dim
    sia file senza header:
        token val1 val2 ...

    Nota: per file molto grandi può richiedere RAM.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    embeddings: dict[str, np.ndarray] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip().split()

        has_header = (
            len(first_line) == 2
            and first_line[0].isdigit()
            and first_line[1].isdigit()
        )

        if has_header:
            dim = int(first_line[1])
        else:
            token = first_line[0]
            vector = np.asarray(first_line[1:], dtype=np.float32)
            dim = vector.shape[0]
            embeddings[token] = vector

        if expected_dim is not None and dim != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {expected_dim}, found {dim}"
            )

        for line in f:
            parts = line.rstrip().split(" ")

            if len(parts) <= 2:
                continue

            token = parts[0]
            values = parts[1:]

            if expected_dim is not None and len(values) != expected_dim:
                continue

            try:
                vector = np.asarray(values, dtype=np.float32)
            except ValueError:
                continue

            if expected_dim is not None and vector.shape[0] != expected_dim:
                continue

            embeddings[token] = vector

    return embeddings


def build_embedding_matrix(
    token_vocab: Vocabulary,
    embedding_config: dict,
    preprocessing_config: dict,
) -> torch.Tensor:
    """Build the word-embedding matrix indexed exactly like ``token_vocab``.

    Known tokens reuse pretrained vectors. Missing tokens receive either a
    deterministic random vector or a zero vector according to the embedding
    config. Centering and L2 normalization exclude ``<PAD>`` because padding
    must remain the all-zero vector consumed by ``nn.Embedding``.
    """
    embedding_cfg = embedding_config.get("embedding", embedding_config)
    preprocessing = preprocessing_config.get("preprocessing", preprocessing_config)

    path = embedding_cfg["path"]
    dim = embedding_cfg["dim"]

    lowercase_lookup = embedding_cfg.get("lowercase_lookup", True)
    oov_strategy = embedding_cfg.get("oov_strategy", "random")
    oov_scale = embedding_cfg.get("oov_scale", 0.05)

    normalize_vectors = preprocessing.get(
        "normalize_embeddings",
        embedding_cfg.get("normalize_vectors", False),
    )

    center_vectors = preprocessing.get("center_embeddings", False)

    pretrained = load_text_embeddings(path, expected_dim=dim)

    matrix = np.zeros((len(token_vocab), dim), dtype=np.float32)

    # Fixed initialization keeps OOV vectors comparable across experiments.
    rng = np.random.default_rng(42)

    found = 0
    missing = 0

    for token, idx in token_vocab.stoi.items():
        if token == PAD_TOKEN:
            matrix[idx] = np.zeros(dim, dtype=np.float32)
            continue

        vector = pretrained.get(token)

        if vector is None and lowercase_lookup:
            vector = pretrained.get(token.lower())

        if vector is not None:
            matrix[idx] = vector
            found += 1
        else:
            missing += 1

            if oov_strategy == "random":
                matrix[idx] = rng.normal(
                    loc=0.0,
                    scale=oov_scale,
                    size=dim,
                ).astype(np.float32)
            elif oov_strategy == "zero":
                matrix[idx] = np.zeros(dim, dtype=np.float32)
            else:
                raise ValueError(f"Unknown oov_strategy: {oov_strategy}")

    if center_vectors:
        non_pad_mask = np.ones(len(token_vocab), dtype=bool)
        non_pad_mask[token_vocab.stoi[PAD_TOKEN]] = False
        mean_vector = matrix[non_pad_mask].mean(axis=0, keepdims=True)
        matrix[non_pad_mask] = matrix[non_pad_mask] - mean_vector

    if normalize_vectors:
        non_pad_mask = np.ones(len(token_vocab), dtype=bool)
        non_pad_mask[token_vocab.stoi[PAD_TOKEN]] = False
        matrix[non_pad_mask] = l2_normalize_matrix(matrix[non_pad_mask])

    pad_idx = token_vocab.stoi[PAD_TOKEN]
    # Re-apply the invariant after optional centering and normalization.
    matrix[pad_idx] = np.zeros(dim, dtype=np.float32)

    coverage = found / max(found + missing, 1)

    print(
        f"Embedding coverage: found={found}, missing={missing}, "
        f"coverage={coverage:.2%}"
    )

    return torch.tensor(matrix, dtype=torch.float32)
