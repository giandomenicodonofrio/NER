"""Read KIND-style TSV files while preserving sentence and domain metadata."""

from pathlib import Path
from dataclasses import dataclass
from src.preprocessing.tokens import normalize_token


@dataclass
class Sentence:
    """One tokenized sentence before conversion to numeric ids."""

    tokens: list[str]
    labels: list[str]
    dataset: str
    source_file: str


def read_ner_tsv(
    path: str | Path,
    dataset_name: str,
    config: dict,
    has_labels: bool = True,
) -> list[Sentence]:
    """Read a TSV split where blank lines delimit sentences.

    The reader keeps the originating dataset name because the same pipeline is
    used for domain-specific metrics and for balanced multi-domain sampling.
    Unlabelled files can be loaded by assigning the neutral ``O`` label.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    preprocessing = config.get("preprocessing", {})
    token_col = preprocessing.get("token_column", 0)
    label_col = preprocessing.get("label_column", -1)

    sentences: list[Sentence] = []
    tokens: list[str] = []
    labels: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line.strip():
                if tokens:
                    sentences.append(
                        Sentence(
                            tokens=tokens,
                            labels=labels,
                            dataset=dataset_name,
                            source_file=str(path),
                        )
                    )
                    tokens = []
                    labels = []
                continue

            parts = line.split("\t")

            token = parts[token_col]
            token = normalize_token(token, config)
            tokens.append(token)

            if has_labels:
                label = parts[label_col]
            else:
                label = "O"

            labels.append(label)

    if tokens:
        sentences.append(
            Sentence(
                tokens=tokens,
                labels=labels,
                dataset=dataset_name,
                source_file=str(path),
            )
        )

    return sentences


def load_datasets_split(
    config: dict,
    split: str,
) -> list[Sentence]:
    """Concatenate the datasets configured for one split."""
    all_sentences: list[Sentence] = []

    dataset_names = config["data"][f"{split}_datasets"]
    dataset_paths = config["data"]["datasets"]

    for dataset_name in dataset_names:
        path = dataset_paths[dataset_name][split]
        sentences = read_ner_tsv(
            path=path,
            dataset_name=dataset_name,
            config=config,
            has_labels=True,
        )
        all_sentences.extend(sentences)

    return all_sentences
