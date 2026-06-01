from collections import Counter
from dataclasses import dataclass
from src.utils.reader import Sentence


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass
class Vocabulary:
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
    labels = sorted({label for sentence in sentences for label in sentence.labels})

    if "O" in labels:
        labels.remove("O")
        labels = ["O"] + labels

    stoi = {label: idx for idx, label in enumerate(labels)}
    return Vocabulary(stoi=stoi, itos=labels)


def build_dataset_vocab(sentences: list[Sentence]) -> Vocabulary:
    datasets = sorted({sentence.dataset for sentence in sentences})
    stoi = {name: idx for idx, name in enumerate(datasets)}
    return Vocabulary(stoi=stoi, itos=datasets)