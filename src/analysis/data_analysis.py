#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd


SUMMARY_EXCLUDED_KEYS = {
    "labels",
    "entities_by_type",
    "entity_lengths",
    "vocab",
    "token_counter",
}


Sentence = list[tuple[str, str]]
Entity = tuple[str, int, int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze NER TSV datasets and generate summary CSV files."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("./dataset"),
        help="Directory containing .tsv files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./outputs/analysis/data_analysis/dataset_insights.csv"),
        help="Output path for the main dataset insights CSV.",
    )
    return parser.parse_args()


def read_ner_tsv(path: Path) -> list[Sentence]:
    """Read a NER TSV file and return sentences as lists of (token, label)."""
    sentences: list[Sentence] = []
    current_sentence: Sentence = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.rstrip("\n")

            if not line.strip():
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            parts = line.split("\t")

            if len(parts) < 2:
                print(f"Warning: malformed line skipped in {path.name}:{line_number}")
                continue

            token = parts[0]
            label = parts[-1]

            if line_number == 1 and token.lower() in {"token", "word"}:
                continue

            current_sentence.append((token, label))

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def close_entity(
    entities: list[Entity],
    entity_type: str | None,
    start: int | None,
    end: int,
    tokens: list[str],
) -> None:
    if entity_type is not None and start is not None:
        entities.append((entity_type, start, end, " ".join(tokens)))


def bio_to_entities(sentence: Sentence) -> list[Entity]:
    """Convert BIO-tagged tokens into entity spans."""
    entities: list[Entity] = []

    current_type: str | None = None
    current_tokens: list[str] = []
    start: int | None = None

    for index, (token, label) in enumerate(sentence):
        if label == "O" or "-" not in label:
            close_entity(
                entities,
                current_type,
                start,
                index - 1,
                current_tokens,
            )
            current_type = None
            current_tokens = []
            start = None
            continue

        prefix, entity_type = label.split("-", 1)

        if prefix == "B" or current_type != entity_type:
            close_entity(
                entities,
                current_type,
                start,
                index - 1,
                current_tokens,
            )
            current_type = entity_type
            current_tokens = [token]
            start = index

        elif prefix == "I":
            current_tokens.append(token)

        else:
            close_entity(
                entities,
                current_type,
                start,
                index - 1,
                current_tokens,
            )
            current_type = None
            current_tokens = []
            start = None

    close_entity(
        entities,
        current_type,
        start,
        len(sentence) - 1,
        current_tokens,
    )

    return entities


def analyze_ner_file(path: Path) -> dict[str, Any]:
    sentences = read_ner_tsv(path)

    label_counts: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    entity_length_counts: Counter[int] = Counter()
    token_counts: Counter[str] = Counter()

    sentence_lengths: list[int] = []
    malformed_i_tags = 0

    for sentence in sentences:
        sentence_lengths.append(len(sentence))
        previous_label = "O"

        for token, label in sentence:
            token_counts[token] += 1
            label_counts[label] += 1

            if label.startswith("I-"):
                entity_type = label[2:]
                valid_previous_labels = {
                    f"B-{entity_type}",
                    f"I-{entity_type}",
                }

                if previous_label not in valid_previous_labels:
                    malformed_i_tags += 1

            previous_label = label

        for entity_type, start, end, _text in bio_to_entities(sentence):
            entity_counts[entity_type] += 1
            entity_length_counts[end - start + 1] += 1

    total_tokens = sum(token_counts.values())
    total_entities = sum(entity_counts.values())

    return {
        "file": path.name,
        "sentences": len(sentences),
        "tokens": total_tokens,
        "unique_tokens": len(token_counts),
        "singleton_tokens": sum(1 for count in token_counts.values() if count == 1),
        "entities": total_entities,
        "avg_sentence_len": (
            sum(sentence_lengths) / len(sentence_lengths)
            if sentence_lengths
            else 0
        ),
        "max_sentence_len": max(sentence_lengths) if sentence_lengths else 0,
        "malformed_i_tags": malformed_i_tags,
        "labels": label_counts,
        "entities_by_type": entity_counts,
        "entity_lengths": entity_length_counts,
        "entity_density": total_entities / max(total_tokens, 1),
        "singleton_ratio": (
            sum(1 for count in token_counts.values() if count == 1)
            / max(len(token_counts), 1)
        ),
        "type_token_ratio": len(token_counts) / max(total_tokens, 1),
        "vocab": set(token_counts.keys()),
        "token_counter": token_counts,
    }


def counter_to_rows(
    file_name: str,
    counter: Counter[Any],
    key_name: str,
    total: int | None = None,
) -> list[dict[str, Any]]:
    denominator = max(total if total is not None else sum(counter.values()), 1)

    return [
        {
            "file": file_name,
            key_name: key,
            "count": count,
            "percentage": count / denominator,
        }
        for key, count in counter.items()
    ]


def summary_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in SUMMARY_EXCLUDED_KEYS
    }


def print_counter(title: str, counter: Counter[Any]) -> None:
    total = max(sum(counter.values()), 1)

    print(f"\n{title}")
    print("-" * len(title))

    for key, value in counter.most_common():
        print(f"{str(key):20s} {value:8d} {value / total:8.2%}")


def print_file_report(result: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f"FILE: {result['file']}")
    print("=" * 80)

    for key, value in summary_row(result).items():
        print(f"{key:20s}: {value}")

    print_counter("Label BIO", result["labels"])
    print_counter("Classi entity", result["entities_by_type"])
    print_counter("Lunghezze entity", result["entity_lengths"])


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(len(a | b), 1)


def vocab_overlap_table(vocabs: dict[str, set[str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    names = sorted(vocabs)

    for dataset_a in names:
        for dataset_b in names:
            vocab_a = vocabs[dataset_a]
            vocab_b = vocabs[dataset_b]
            intersection = vocab_a & vocab_b
            union = vocab_a | vocab_b

            rows.append(
                {
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "intersection": len(intersection),
                    "union": len(union),
                    "jaccard": len(intersection) / max(len(union), 1),
                    "coverage_a_in_b": len(intersection) / max(len(vocab_a), 1),
                    "coverage_b_in_a": len(intersection) / max(len(vocab_b), 1),
                }
            )

    return pd.DataFrame(rows)


def save_vocab_overlap_heatmap(
    overlap_df: pd.DataFrame,
    out_dir: Path,
    value_col: str = "jaccard",
) -> Path:
    if value_col not in overlap_df.columns:
        raise ValueError(f"Column not found in overlap dataframe: {value_col}")

    matrix = overlap_df.pivot(
        index="dataset_a",
        columns="dataset_b",
        values=value_col,
    )

    row_labels = matrix.index.tolist()
    col_labels = matrix.columns.tolist()
    values = matrix.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(values, vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    ax.set_title(f"Vocabulary overlap - {value_col}")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            label = "NA" if pd.isna(value) else f"{value:.2f}"

            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
            )

    fig.tight_layout()

    out_path = out_dir / f"vocab_overlap_{value_col}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path


def save_vocab_overlap_dual_heatmap(
    overlap_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    a_in_b = overlap_df.pivot(
        index="dataset_a",
        columns="dataset_b",
        values="coverage_a_in_b",
    )

    b_in_a = overlap_df.pivot(
        index="dataset_a",
        columns="dataset_b",
        values="coverage_b_in_a",
    )

    row_labels = a_in_b.index.tolist()
    col_labels = a_in_b.columns.tolist()

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.set_xlim(0, len(col_labels))
    ax.set_ylim(0, len(row_labels))
    ax.invert_yaxis()
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(len(col_labels)) + 0.5)
    ax.set_yticks(np.arange(len(row_labels)) + 0.5)
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    ax.set_title("Vocabulary overlap: A in B / B in A")

    cmap = plt.colormaps["viridis"]

    for row_idx, dataset_a in enumerate(row_labels):
        for col_idx, dataset_b in enumerate(col_labels):
            value_a_in_b = a_in_b.loc[dataset_a, dataset_b]
            value_b_in_a = b_in_a.loc[dataset_a, dataset_b]

            top_triangle = Polygon(
                [
                    (col_idx, row_idx),          # alto-sinistra
                    (col_idx + 1, row_idx),      # alto-destra
                    (col_idx, row_idx + 1),      # basso-sinistra
                ],
                facecolor=cmap(value_a_in_b),
                edgecolor="white",
            )

            bottom_triangle = Polygon(
                [
                    (col_idx + 1, row_idx),      # alto-destra
                    (col_idx, row_idx + 1),      # basso-sinistra
                    (col_idx + 1, row_idx + 1),  # basso-destra
                ],
                facecolor=cmap(value_b_in_a),
                edgecolor="white",
            )

            ax.add_patch(top_triangle)
            ax.add_patch(bottom_triangle)

            # testo nel triangolo alto-sinistra
            ax.text(
                col_idx + 0.30,
                row_idx + 0.30,
                f"{value_a_in_b:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

            # testo nel triangolo basso-destra
            ax.text(
                col_idx + 0.70,
                row_idx + 0.70,
                f"{value_b_in_a:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(0, 1)
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("dataset_b")
    ax.set_ylabel("dataset_a")

    fig.tight_layout()

    out_path = out_dir / "vocab_overlap_dual_coverage.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path

def save_top_tokens(
    token_counters: dict[str, Counter[str]],
    out_dir: Path,
    n: int = 100,
) -> Path:
    rows: list[dict[str, Any]] = []

    for dataset_name, counter in token_counters.items():
        total = max(sum(counter.values()), 1)

        for token, count in counter.most_common(n):
            rows.append(
                {
                    "dataset": dataset_name,
                    "token": token,
                    "count": count,
                    "frequency": count / total,
                }
            )

    out_path = out_dir / "top_tokens.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    return out_path


def save_outputs(
    summary_rows: list[dict[str, Any]],
    entity_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    token_counters: dict[str, Counter[str]],
    vocabs: dict[str, set[str]],
    out_path: Path,
) -> None:
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    top_tokens_path = save_top_tokens(token_counters, out_dir)
    print(f"Top token salvati in {top_tokens_path}")

    entities_path = out_dir / "entities_by_type.csv"
    pd.DataFrame(entity_rows).to_csv(entities_path, index=False)
    print(f"Distribuzione entity salvata in {entities_path}")

    labels_path = out_dir / "labels_distribution.csv"
    pd.DataFrame(label_rows).to_csv(labels_path, index=False)
    print(f"Distribuzione label salvata in {labels_path}")

    pd.DataFrame(summary_rows).to_csv(out_path, index=False)
    print(f"CSV principale salvato in {out_path}")

    if not vocabs:
        print("Nessun vocabolario trovato: salto overlap e heatmap.")
        return

    overlap_df = vocab_overlap_table(vocabs)

    overlap_path = out_dir / "vocab_overlap.csv"
    overlap_df.to_csv(overlap_path, index=False)
    print(f"Overlap vocabolario salvato in {overlap_path}")

    dual_heatmap_path = save_vocab_overlap_dual_heatmap(overlap_df, out_dir)
    print(f"Heatmap doppia overlap salvata in {dual_heatmap_path}")


def main() -> None:
    args = parse_args()

    files = sorted(args.dataset_dir.rglob("*.tsv"))

    if not files:
        raise FileNotFoundError(f"Nessun file .tsv trovato in {args.dataset_dir}")

    summary_rows: list[dict[str, Any]] = []
    entity_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []

    global_labels: Counter[str] = Counter()
    global_entities: Counter[str] = Counter()
    global_entity_lengths: Counter[int] = Counter()

    vocabs: dict[str, set[str]] = {}
    token_counters: dict[str, Counter[str]] = {}

    for file_path in files:
        result = analyze_ner_file(file_path)

        split_name = file_path.stem

        vocabs[split_name] = result["vocab"]
        token_counters[split_name] = result["token_counter"]

        summary_rows.append(summary_row(result))

        entity_rows.extend(
            counter_to_rows(
                file_name=file_path.name,
                counter=result["entities_by_type"],
                key_name="entity_type",
            )
        )

        label_rows.extend(
            counter_to_rows(
                file_name=file_path.name,
                counter=result["labels"],
                key_name="label",
            )
        )

        global_labels.update(result["labels"])
        global_entities.update(result["entities_by_type"])
        global_entity_lengths.update(result["entity_lengths"])

        print_file_report(result)

    save_outputs(
        summary_rows=summary_rows,
        entity_rows=entity_rows,
        label_rows=label_rows,
        token_counters=token_counters,
        vocabs=vocabs,
        out_path=args.out,
    )

    print_counter("Label BIO globali", global_labels)
    print_counter("Classi entity globali", global_entities)
    print_counter("Lunghezze entity globali", global_entity_lengths)


if __name__ == "__main__":
    main()