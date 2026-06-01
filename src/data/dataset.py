from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src.utils.reader import Sentence
from src.utils.vocabulary import Vocabulary


@dataclass
class EncodedSentence:
    token_ids: list[int]
    char_ids: list[list[int]]
    label_ids: list[int]
    dataset_id: int
    tokens: list[str]
    labels: list[str]
    dataset: str


class NERDataset(Dataset):
    def __init__(
        self,
        sentences: list[Sentence],
        token_vocab: Vocabulary,
        char_vocab: Vocabulary,
        label_vocab: Vocabulary,
        dataset_vocab: Vocabulary,
        remove_stopwords: bool = False,
        stopwords: set[str] | None = None,
        fix_malformed_i_tags: bool = False,
    ):
        self.sentences = sentences
        self.token_vocab = token_vocab
        self.char_vocab = char_vocab
        self.label_vocab = label_vocab
        self.dataset_vocab = dataset_vocab
        self.remove_stopwords = remove_stopwords
        self.stopwords = stopwords or set()
        self.fix_malformed_i_tags = fix_malformed_i_tags

        self.encoded = [self.encode_sentence(sentence) for sentence in sentences]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> EncodedSentence:
        return self.encoded[idx]

    def encode_sentence(self, sentence: Sentence) -> EncodedSentence:
        tokens = []
        labels = []

        for token, label in zip(sentence.tokens, sentence.labels):
            if self.remove_stopwords and token.lower() in self.stopwords:
                continue

            if self.fix_malformed_i_tags and label.startswith("I-"):
                entity_type = label[2:]
                valid_previous_labels = {
                    f"B-{entity_type}",
                    f"I-{entity_type}",
                }

                if not labels or labels[-1] not in valid_previous_labels:
                    label = f"B-{entity_type}"

            tokens.append(token)
            labels.append(label)

        if not tokens:
            tokens = ["<EMPTY>"]
            labels = ["O"]

        token_ids = [self.token_vocab.encode(token) for token in tokens]

        char_ids = [
            [self.char_vocab.encode(char) for char in token]
            for token in tokens
        ]

        label_ids = [self.label_vocab.stoi[label] for label in labels]
        dataset_id = self.dataset_vocab.stoi[sentence.dataset]

        return EncodedSentence(
            token_ids=token_ids,
            char_ids=char_ids,
            label_ids=label_ids,
            dataset_id=dataset_id,
            tokens=tokens,
            labels=labels,
            dataset=sentence.dataset,
        )


def collate_ner_batch(batch: list[EncodedSentence]) -> dict[str, torch.Tensor | list]:
    batch_size = len(batch)
    max_seq_len = max(len(item.token_ids) for item in batch)
    max_word_len = max(
        max(len(chars) for chars in item.char_ids)
        for item in batch
    )

    token_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    label_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

    char_ids = torch.zeros(batch_size, max_seq_len, max_word_len, dtype=torch.long)
    char_mask = torch.zeros(batch_size, max_seq_len, max_word_len, dtype=torch.bool)

    dataset_ids = torch.zeros(batch_size, dtype=torch.long)

    tokens = []
    labels = []
    datasets = []

    for i, item in enumerate(batch):
        seq_len = len(item.token_ids)
        dataset_ids[i] = item.dataset_id

        token_ids[i, :seq_len] = torch.tensor(item.token_ids, dtype=torch.long)
        label_ids[i, :seq_len] = torch.tensor(item.label_ids, dtype=torch.long)
        mask[i, :seq_len] = True

        for j, word_chars in enumerate(item.char_ids):
            word_len = len(word_chars)
            char_ids[i, j, :word_len] = torch.tensor(word_chars, dtype=torch.long)
            char_mask[i, j, :word_len] = True

        tokens.append(item.tokens)
        labels.append(item.labels)
        datasets.append(item.dataset)

    return {
        "token_ids": token_ids,
        "char_ids": char_ids,
        "label_ids": label_ids,
        "mask": mask,
        "char_mask": char_mask,
        "dataset_ids": dataset_ids,
        "tokens": tokens,
        "labels": labels,
        "datasets": datasets,
    }
