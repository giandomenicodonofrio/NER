from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import json
from pathlib import Path

import pandas as pd
import torch

from src.utils.config import load_experiment_config
from src.data.datamodule import build_datasets_and_vocabs, build_dataloaders
from src.preprocessing.embeddings import build_embedding_matrix
from src.models.sequence_tagger import SequenceTagger
from src.training.trainer import Trainer


EVALUATIONS = [
    {
        "name": "all_datasets_model",
        "config": "configs/experiment/post_tuning/post_tuning_all_datasets_word_dropout_010.yaml",
        "checkpoint": "outputs/post_tuning/post_tuning_all_datasets_word_dropout_010/checkpoints/best.pt",
        "output_dir": "outputs/final_evaluation/all_datasets_model",
    },
    {
        "name": "wn_model",
        "config": "configs/experiment/tuning/tuning_wn_word_dropout_010.yaml",
        "checkpoint": "outputs/tuning/tuning_wn_word_dropout_010/checkpoints/best.pt",
        "output_dir": "outputs/final_evaluation/wn_model",
    },
]


ALL_TEST_DATASETS = {
    "ADG": {
        "train": "data/raw/ADG/ADG_train.tsv",
        "dev": "data/raw/ADG/ADG_dev.tsv",
        "test": "data/raw/ADG/ADG_test.tsv",
    },
    "FIC": {
        "train": "data/raw/FIC/FIC_train.tsv",
        "dev": "data/raw/FIC/FIC_dev.tsv",
        "test": "data/raw/FIC/FIC_test.tsv",
    },
    "WN": {
        "train": "data/raw/WN/WN_train.tsv",
        "dev": "data/raw/WN/WN_dev.tsv",
        "test": "data/raw/WN/WN_test.tsv",
    },
}


def patch_config_for_all_tests(config: dict, output_dir: str) -> dict:
    config["experiment"]["output_dir"] = output_dir

    config["data"]["datasets"] = ALL_TEST_DATASETS
    config["data"]["test_datasets"] = ["ADG", "FIC", "WN"]

    return config


def evaluate_model(eval_cfg: dict, device: str | None = None) -> list[dict]:
    config = load_experiment_config(eval_cfg["config"])
    config = patch_config_for_all_tests(config, eval_cfg["output_dir"])

    output_dir = Path(eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets, vocabs = build_datasets_and_vocabs(config)
    dataloaders = build_dataloaders(config, datasets)

    embedding_matrix = build_embedding_matrix(
        token_vocab=vocabs["token"],
        embedding_config=config["embedding"],
        preprocessing_config=config["preprocessing"],
    )

    model = SequenceTagger(
        embedding_matrix=embedding_matrix,
        num_chars=len(vocabs["char"]),
        num_labels=len(vocabs["label"]),
        config=config,
        pad_token_id=vocabs["token"].stoi["<PAD>"],
        pad_char_id=vocabs["char"].stoi["<PAD>"],
    )

    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        config=config,
        label_vocab=vocabs["label"],
        device=device,
    )

    checkpoint = torch.load(
        eval_cfg["checkpoint"],
        map_location=trainer.device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = trainer.evaluate(split="test", save_outputs=True)

    metrics_by_dataset_path = output_dir / "metrics" / "test_metrics_by_dataset.json"

    with metrics_by_dataset_path.open("r", encoding="utf-8") as f:
        metrics_by_dataset = json.load(f)

    rows = []

    rows.append(
        {
            "model": eval_cfg["name"],
            "dataset": "ALL",
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "token_accuracy": metrics["token_accuracy"],
        }
    )

    for dataset_name, dataset_metrics in metrics_by_dataset.items():
        rows.append(
            {
                "model": eval_cfg["name"],
                "dataset": dataset_name,
                "precision": dataset_metrics["precision"],
                "recall": dataset_metrics["recall"],
                "f1": dataset_metrics["f1"],
                "token_accuracy": dataset_metrics["token_accuracy"],
            }
        )

    return rows


def main():
    all_rows = []

    for eval_cfg in EVALUATIONS:
        print(f"\nEvaluating: {eval_cfg['name']}")
        rows = evaluate_model(eval_cfg, device="cuda" if torch.cuda.is_available() else "cpu")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    output_dir = Path("outputs/final_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "final_comparison.csv", index=False)

    print("\n=== Final comparison ===")
    print(df.to_string(index=False))

    pivot = df.pivot(
        index="dataset",
        columns="model",
        values="f1",
    )

    pivot.to_csv(output_dir / "final_comparison_f1_pivot.csv")

    print("\n=== F1 pivot ===")
    print(pivot.to_string())


if __name__ == "__main__":
    main() 