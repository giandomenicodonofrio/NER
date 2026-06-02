"""Build datasets, vocabularies and dataloaders from an experiment config."""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from collections import Counter
from pathlib import Path

from src.data.dataset import NERDataset, collate_ner_batch
from src.utils.reader import load_datasets_split
from src.utils.vocabulary import (
    build_token_vocab,
    build_char_vocab,
    build_label_vocab,
    build_dataset_vocab,
)

def build_dataset_balanced_sampler(dataset):
    """Sample domains uniformly while preserving the epoch length.

    Each sentence receives the inverse frequency of its source domain. With
    replacement enabled, the sampler draws the same number of examples as a
    normal epoch but prevents the largest domain from dominating training.
    """
    dataset_names = [item.dataset for item in dataset.encoded]

    counts = Counter(dataset_names)

    weights = [
        1.0 / counts[name]
        for name in dataset_names
    ]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )

    return sampler

def load_stopwords(path: str | None) -> set[str]:
    """Load one lowercase stopword per line, ignoring blank lines and comments."""
    if path is None:
        return set()

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Stopwords file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return {
            line.strip().lower()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        }


def build_datasets_and_vocabs(config: dict):
    """Construct encoded splits and the vocabularies shared by all splits.

    Token, character and label vocabularies are learned only from training
    sentences to avoid lexical leakage from dev and test. The dataset vocabulary
    may inspect every split because domain ids are metadata used for reporting
    and sampling, not model features.
    """
    preprocessing = config.get("preprocessing", {})
    remove_stopwords = preprocessing.get("remove_stopwords", False)
    stopwords_path = preprocessing.get("stopwords_path")

    if remove_stopwords and not stopwords_path:
        raise ValueError(
            "remove_stopwords=true requires preprocessing.stopwords_path"
        )

    stopwords = load_stopwords(stopwords_path) if remove_stopwords else set()

    if remove_stopwords and not stopwords:
        raise ValueError(f"Stopwords file is empty: {stopwords_path}")

    train_sentences = load_datasets_split(config, "train")
    dev_sentences = load_datasets_split(config, "dev")
    test_sentences = load_datasets_split(config, "test")

    min_token_freq = preprocessing.get("min_token_freq", 1)
    min_char_freq = preprocessing.get("min_char_freq", 1)
    # BIO repair is only a consequence of token removal. Preserve source gold
    # labels unchanged for the baseline, development and held-out test data.
    fix_malformed_i_tags = (
        remove_stopwords
        and preprocessing.get("fix_malformed_i_tags", False)
    )

    token_vocab = build_token_vocab(train_sentences, min_freq=min_token_freq)
    char_vocab = build_char_vocab(train_sentences, min_freq=min_char_freq)
    label_vocab = build_label_vocab(train_sentences)
    dataset_vocab = build_dataset_vocab(train_sentences + dev_sentences + test_sentences)

    train_dataset = NERDataset(
        train_sentences,
        token_vocab,
        char_vocab,
        label_vocab,
        dataset_vocab,
        remove_stopwords=remove_stopwords,
        stopwords=stopwords,
        fix_malformed_i_tags=fix_malformed_i_tags,
    )

    dev_dataset = NERDataset(
        dev_sentences,
        token_vocab,
        char_vocab,
        label_vocab,
        dataset_vocab,
        remove_stopwords=remove_stopwords,
        stopwords=stopwords,
        fix_malformed_i_tags=fix_malformed_i_tags,
    )

    test_dataset = NERDataset(
        test_sentences,
        token_vocab,
        char_vocab,
        label_vocab,
        dataset_vocab,
        remove_stopwords=remove_stopwords,
        stopwords=stopwords,
        fix_malformed_i_tags=fix_malformed_i_tags,
    )

    vocabs = {
        "token": token_vocab,
        "char": char_vocab,
        "label": label_vocab,
        "dataset": dataset_vocab,
    }

    datasets = {
        "train": train_dataset,
        "dev": dev_dataset,
        "test": test_dataset,
    }

    return datasets, vocabs


def build_dataloaders(config: dict, datasets: dict):
    """Create loaders and apply the configured training sampling strategy."""
    training = config.get("training", {})
    sampling = config.get("sampling", {})

    batch_size = training.get("batch_size", 32)
    shuffle_train = sampling.get("shuffle", True)

    sampling_strategy = sampling.get("strategy", "random")

    if sampling_strategy == "balanced_by_dataset":
        train_sampler = build_dataset_balanced_sampler(datasets["train"])
        # PyTorch does not allow shuffle and sampler to be active together.
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = shuffle_train

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        collate_fn=collate_ner_batch,
    )

    dev_loader = DataLoader(
        datasets["dev"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ner_batch,
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ner_batch,
    )

    return {
        "train": train_loader,
        "dev": dev_loader,
        "test": test_loader,
    }
