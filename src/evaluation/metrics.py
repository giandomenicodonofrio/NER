"""Compute NER metrics and persist evaluation diagnostics."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix


def ids_to_labels(sequences, label_vocab):
    """Decode numeric model outputs into BIO label sequences."""
    return [
        [label_vocab.itos[idx] for idx in seq]
        for seq in sequences
    ]


def compute_entity_metrics(y_true_labels, y_pred_labels) -> dict:
    """Compute span-level NER metrics using seqeval's BIO interpretation."""
    return {
        "entity_precision": precision_score(y_true_labels, y_pred_labels),
        "entity_recall": recall_score(y_true_labels, y_pred_labels),
        "entity_f1": f1_score(y_true_labels, y_pred_labels),
        "entity_report": classification_report(
            y_true_labels,
            y_pred_labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def compute_token_accuracy(y_true_labels, y_pred_labels) -> float:
    """Compute token accuracy as a secondary metric.

    Entity-level F1 remains the primary selection metric because the frequent
    ``O`` label can make token accuracy look strong even for weak NER models.
    """
    correct = 0
    total = 0

    for gold_seq, pred_seq in zip(y_true_labels, y_pred_labels):
        for gold, pred in zip(gold_seq, pred_seq):
            total += 1
            if gold == pred:
                correct += 1

    return correct / total if total else 0.0


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(make_json_serializable(v) for v in obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def save_json(data: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = make_json_serializable(data)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_predictions_tsv(
    tokens,
    gold_labels,
    pred_labels,
    datasets,
    path: str | Path,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("dataset\ttoken\tgold\tpred\tis_error\n")

        for sent_tokens, sent_gold, sent_pred, dataset in zip(
            tokens, gold_labels, pred_labels, datasets
        ):
            for token, gold, pred in zip(sent_tokens, sent_gold, sent_pred):
                is_error = int(gold != pred)
                f.write(f"{dataset}\t{token}\t{gold}\t{pred}\t{is_error}\n")

            f.write("\n")


def save_errors_tsv(
    tokens,
    gold_labels,
    pred_labels,
    datasets,
    path: str | Path,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("dataset\ttoken\tgold\tpred\n")

        for sent_tokens, sent_gold, sent_pred, dataset in zip(
            tokens, gold_labels, pred_labels, datasets
        ):
            for token, gold, pred in zip(sent_tokens, sent_gold, sent_pred):
                if gold != pred:
                    f.write(f"{dataset}\t{token}\t{gold}\t{pred}\n")

def save_confusion_matrix(
    y_true_labels,
    y_pred_labels,
    labels,
    output_dir: str | Path,
):
    """Save raw and normalized token-level confusion matrices.

    These matrices diagnose label confusions but do not replace span-level
    seqeval metrics, which remain the primary NER evaluation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flat_true = [x for seq in y_true_labels for x in seq]
    flat_pred = [x for seq in y_pred_labels for x in seq]

    cm = confusion_matrix(flat_true, flat_pred, labels=labels)

    np.save(output_dir / "confusion_raw.npy", cm)

    df_raw = pd.DataFrame(cm, index=labels, columns=labels)
    df_raw.to_csv(output_dir / "confusion_raw.csv")

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    df_norm = pd.DataFrame(cm_norm, index=labels, columns=labels)
    df_norm.to_csv(output_dir / "confusion_normalized.csv")

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm_norm[i, j]
            if value > 0:
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title("Token-level Confusion Matrix - Row Normalized")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_normalized.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(np.log1p(cm))

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title("Token-level Confusion Matrix - log(1 + count)")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_log_counts.png", dpi=200)
    plt.close(fig)

def compute_metrics_by_dataset(
    tokens,
    y_true_labels,
    y_pred_labels,
    datasets,
) -> dict:
    """Compute the same metrics separately for each source domain."""
    grouped = {}

    for token_seq, gold_seq, pred_seq, dataset in zip(
        tokens,
        y_true_labels,
        y_pred_labels,
        datasets,
    ):
        if dataset not in grouped:
            grouped[dataset] = {
                "tokens": [],
                "gold": [],
                "pred": [],
            }

        grouped[dataset]["tokens"].append(token_seq)
        grouped[dataset]["gold"].append(gold_seq)
        grouped[dataset]["pred"].append(pred_seq)

    results = {}

    for dataset, values in grouped.items():
        entity_metrics = compute_entity_metrics(
            values["gold"],
            values["pred"],
        )

        token_accuracy = compute_token_accuracy(
            values["gold"],
            values["pred"],
        )

        results[dataset] = {
            "precision": entity_metrics["entity_precision"],
            "recall": entity_metrics["entity_recall"],
            "f1": entity_metrics["entity_f1"],
            "token_accuracy": token_accuracy,
            "num_sentences": len(values["gold"]),
            "num_tokens": sum(len(seq) for seq in values["gold"]),
        }

    return results
