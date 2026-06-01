"""Training loop, checkpoint selection and evaluation orchestration."""

from pathlib import Path
import json

import torch
from torch.optim import AdamW
from tqdm import tqdm

from src.evaluation.metrics import (
    ids_to_labels,
    compute_entity_metrics,
    compute_token_accuracy,
    save_json,
    save_predictions_tsv,
    save_errors_tsv,
    save_confusion_matrix,
    compute_metrics_by_dataset,
)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move tensors to the target device while preserving textual metadata."""
    moved = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value

    return moved


def flatten_predictions(predictions, labels, mask):
    y_true = []
    y_pred = []

    labels = labels.detach().cpu().tolist()
    mask = mask.detach().cpu().tolist()

    for pred_seq, gold_seq, mask_seq in zip(predictions, labels, mask):
        valid_len = sum(mask_seq)

        y_pred.extend(pred_seq[:valid_len])
        y_true.extend(gold_seq[:valid_len])

    return y_true, y_pred


def token_level_scores(y_true, y_pred, o_label_id: int = 0) -> dict:
    correct = 0
    total = 0

    entity_correct = 0
    entity_pred_total = 0
    entity_gold_total = 0

    for gold, pred in zip(y_true, y_pred):
        if gold == pred:
            correct += 1

        total += 1

        if pred != o_label_id:
            entity_pred_total += 1

        if gold != o_label_id:
            entity_gold_total += 1

        if gold == pred and gold != o_label_id:
            entity_correct += 1

    accuracy = correct / total if total else 0.0

    precision = (
        entity_correct / entity_pred_total
        if entity_pred_total
        else 0.0
    )

    recall = (
        entity_correct / entity_gold_total
        if entity_gold_total
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "token_accuracy": accuracy,
        "token_entity_precision": precision,
        "token_entity_recall": recall,
        "token_entity_f1": f1,
    }


class Trainer:
    """Train a tagger and select its best checkpoint using development F1."""

    def __init__(
        self,
        model,
        dataloaders: dict,
        config: dict,
        label_vocab,
        device: str | None = None,
    ):
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        self.label_vocab = label_vocab

        self.device = torch.device(
            device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        training_cfg = config.get("training", {})

        self.epochs = training_cfg.get("epochs", 50)
        self.learning_rate = training_cfg.get("learning_rate", 1e-3)
        self.weight_decay = training_cfg.get("weight_decay", 1e-4)
        self.gradient_clip_norm = training_cfg.get("gradient_clip_norm", 5.0)
        self.early_stopping_patience = training_cfg.get("early_stopping_patience", 8)

        self.output_dir = Path(config["experiment"]["output_dir"])
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.metrics_dir = self.output_dir / "metrics"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.best_score = -1.0
        self.best_epoch = -1
        self.bad_epochs = 0

    def train(self):
        """Train until all epochs run or development F1 stops improving."""
        history = []

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            dev_metrics = self.evaluate(split="dev")

            score = dev_metrics["f1"]

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"dev_{k}": v for k, v in dev_metrics.items()},
            }

            history.append(row)

            print(
                f"Epoch {epoch:03d} | "
                f"loss={train_loss:.4f} | "
                f"dev_f1={score:.4f}"
            )

            if score > self.best_score:
                # Keep model selection isolated from the held-out test split.
                self.best_score = score
                self.best_epoch = epoch
                self.bad_epochs = 0
                self.save_checkpoint("best.pt")
            else:
                self.bad_epochs += 1

            self.save_history(history)

            if self.bad_epochs >= self.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch: {self.best_epoch}, best score: {self.best_score:.4f}"
                )
                break

        return history

    def train_one_epoch(self, epoch: int) -> float:
        """Run one optimization epoch and return the mean batch loss."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        progress = tqdm(
            self.dataloaders["train"],
            desc=f"Train epoch {epoch}",
            leave=False,
        )

        for batch in progress:
            batch = move_batch_to_device(batch, self.device)

            self.optimizer.zero_grad()

            loss = self.model(
                token_ids=batch["token_ids"],
                char_ids=batch["char_ids"],
                char_mask=batch["char_mask"],
                mask=batch["mask"],
                labels=batch["label_ids"],
            )

            loss.backward()

            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm,
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            progress.set_postfix(loss=loss.item())

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, split: str = "dev", save_outputs: bool = False) -> dict:
        """Compute entity-level metrics and optionally persist diagnostics.

        Saved diagnostics include aggregate and per-domain metrics, a seqeval
        classification report, token-level predictions, errors and confusion
        matrices. Text metadata stays on CPU and is used only when exporting.
        """
        self.model.eval()

        all_gold_ids = []
        all_pred_ids = []

        all_tokens = []
        all_datasets = []

        for batch in tqdm(
            self.dataloaders[split],
            desc=f"Evaluate {split}",
            leave=False,
        ):
            batch = move_batch_to_device(batch, self.device)

            predictions = self.model.decode(
                token_ids=batch["token_ids"],
                char_ids=batch["char_ids"],
                char_mask=batch["char_mask"],
                mask=batch["mask"],
            )

            labels = batch["label_ids"].detach().cpu().tolist()
            mask = batch["mask"].detach().cpu().tolist()

            for pred_seq, gold_seq, mask_seq, token_seq, dataset in zip(
                predictions,
                labels,
                mask,
                batch["tokens"],
                batch["datasets"],
            ):
                valid_len = sum(mask_seq)

                all_pred_ids.append(pred_seq[:valid_len])
                all_gold_ids.append(gold_seq[:valid_len])
                all_tokens.append(token_seq[:valid_len])
                all_datasets.append(dataset)

        y_true_labels = ids_to_labels(all_gold_ids, self.label_vocab)
        y_pred_labels = ids_to_labels(all_pred_ids, self.label_vocab)

        entity_metrics = compute_entity_metrics(y_true_labels, y_pred_labels)
        token_accuracy = compute_token_accuracy(y_true_labels, y_pred_labels)

        metrics = {
            "precision": entity_metrics["entity_precision"],
            "recall": entity_metrics["entity_recall"],
            "f1": entity_metrics["entity_f1"],
            "token_accuracy": token_accuracy,
        }

        metrics_by_dataset = compute_metrics_by_dataset(
            tokens=all_tokens,
            y_true_labels=y_true_labels,
            y_pred_labels=y_pred_labels,
            datasets=all_datasets,
        )

        if save_outputs:
            metrics_dir = self.output_dir / "metrics"
            predictions_dir = self.output_dir / "predictions"
            confusion_dir = self.output_dir / "confusion_matrices"

            save_json(
                metrics_by_dataset,
                metrics_dir / f"{split}_metrics_by_dataset.json",
            )

            save_json(metrics, metrics_dir / f"{split}_metrics.json")
            save_json(
                entity_metrics["entity_report"],
                metrics_dir / f"{split}_classification_report.json",
            )

            save_predictions_tsv(
                tokens=all_tokens,
                gold_labels=y_true_labels,
                pred_labels=y_pred_labels,
                datasets=all_datasets,
                path=predictions_dir / f"{split}_predictions.tsv",
            )

            save_errors_tsv(
                tokens=all_tokens,
                gold_labels=y_true_labels,
                pred_labels=y_pred_labels,
                datasets=all_datasets,
                path=predictions_dir / f"{split}_errors.tsv",
            )

            save_confusion_matrix(
                y_true_labels=y_true_labels,
                y_pred_labels=y_pred_labels,
                labels=self.label_vocab.itos,
                output_dir=confusion_dir / split,
            )

        return metrics

    def save_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_score": self.best_score,
                "best_epoch": self.best_epoch,
                "config": self.config,
                "label_vocab": self.label_vocab.itos,
            },
            path,
        )

    def save_history(self, history: list[dict]):
        path = self.metrics_dir / "history.json"

        with path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
