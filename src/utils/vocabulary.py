"""Build the vocabularies shared by dataset encoding and model decoding."""

from collections import Counter
from dataclasses import dataclass
from src.utils.reader import Sentence


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass
class Vocabulary:
    """Bidirectional mapping between textual items and integer ids."""

    stoi: dict[str, int]
    itos: list[str]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, item: str) -> int:
        return self.stoi.get(item, self.stoi[UNK_TOKEN])

    def decode(self, idx: int) -> str:
        return self.itos[idx]


def build_token_vocab(
    sentences: list[Sentence],
    min_freq: int = 1,
) -> Vocabulary:
    """Build the token vocabulary from training sentences."""
    counter = Counter()

    for sentence in sentences:
        counter.update(sentence.tokens)

    itos = [PAD_TOKEN, UNK_TOKEN]

    for token, freq in counter.items():
        if freq >= min_freq:
            itos.append(token)

    stoi = {token: idx for idx, token in enumerate(itos)}
    return Vocabulary(stoi=stoi, itos=itos)


def build_char_vocab(
    sentences: list[Sentence],
    min_freq: int = 1,
) -> Vocabulary:
    """Build the character vocabulary used by the CharCNN branch."""
    counter = Counter()

    for sentence in sentences:
        for token in sentence.tokens:
            counter.update(token)

    itos = [PAD_TOKEN, UNK_TOKEN]

    for char, freq in counter.items():
        if freq >= min_freq:
            itos.append(char)

    stoi = {char: idx for idx, char in enumerate(itos)}
    return Vocabulary(stoi=stoi, itos=itos)


def build_label_vocab(sentences: list[Sentence]) -> Vocabulary:
    """Build the output vocabulary, keeping ``O`` at index zero.

    ``SequenceTagger`` relies on this invariant when it optionally assigns a
    different auxiliary loss weight to entity tokens and non-entity tokens.
    """
    labels = sorted({label for sentence in sentences for label in sentence.labels})

    if "O" in labels:
        labels.remove("O")
        labels = ["O"] + labels

    stoi = {label: idx for idx, label in enumerate(labels)}
    return Vocabulary(stoi=stoi, itos=labels)


def build_dataset_vocab(sentences: list[Sentence]) -> Vocabulary:
    """Map domain names such as ADG, FIC and WN to ids."""
    datasets = sorted({sentence.dataset for sentence in sentences})
    stoi = {name: idx for idx, name in enumerate(datasets)}
    return Vocabulary(stoi=stoi, itos=datasets)
