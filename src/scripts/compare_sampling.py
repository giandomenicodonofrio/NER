import json
from pathlib import Path

import pandas as pd


EXPERIMENTS = {
    "random": Path("outputs/final/final_all_datasets"),
    "balanced": Path("outputs/balancing/all_balanced_sampling"),
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_global_metrics(exp_dir: Path) -> dict:
    return load_json(exp_dir / "metrics" / "test_metrics.json")


def load_dataset_metrics(exp_dir: Path) -> dict:
    return load_json(exp_dir / "metrics" / "test_metrics_by_dataset.json")


def main():
    rows_global = []
    rows_by_dataset = []

    for setup_name, exp_dir in EXPERIMENTS.items():
        global_metrics = load_global_metrics(exp_dir)
        dataset_metrics = load_dataset_metrics(exp_dir)

        rows_global.append(
            {
                "setup": setup_name,
                "precision": global_metrics["precision"],
                "recall": global_metrics["recall"],
                "f1": global_metrics["f1"],
                "token_accuracy": global_metrics["token_accuracy"],
            }
        )

        for dataset_name, metrics in dataset_metrics.items():
            rows_by_dataset.append(
                {
                    "setup": setup_name,
                    "dataset": dataset_name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "token_accuracy": metrics["token_accuracy"],
                    "num_sentences": metrics["num_sentences"],
                    "num_tokens": metrics["num_tokens"],
                }
            )

    df_global = pd.DataFrame(rows_global)
    df_by_dataset = pd.DataFrame(rows_by_dataset)

    output_dir = Path("outputs/analysis/sampling_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_global.to_csv(output_dir / "sampling_global_metrics.csv", index=False)
    df_by_dataset.to_csv(output_dir / "sampling_by_dataset_metrics.csv", index=False)

    pivot = df_by_dataset.pivot(
        index="dataset",
        columns="setup",
        values="f1",
    )

    pivot["delta_balanced_minus_random"] = (
        pivot["balanced"] - pivot["random"]
    )

    pivot.to_csv(output_dir / "sampling_f1_delta_by_dataset.csv")

    print("\n=== Global metrics ===")
    print(df_global.to_string(index=False))

    print("\n=== Metrics by dataset ===")
    print(df_by_dataset.to_string(index=False))

    print("\n=== F1 delta by dataset ===")
    print(pivot.to_string())

    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()