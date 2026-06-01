from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("outputs/analysis/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_barplot(df: pd.DataFrame, x: str, y: str, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(df[x], df[y])
    ax.set_title(title)
    ax.set_ylabel(y)
    ax.set_ylim(0, max(df[y]) + 0.05)

    for i, value in enumerate(df[y]):
        ax.text(i, value + 0.005, f"{value:.4f}", ha="center")

    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()

    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved: {path}")


def main():
    # 1. Ablazione architettura
    architecture = pd.DataFrame(
        {
            "Model": [
                "BiLSTM + Softmax",
                "BiLSTM + CRF",
                "CharCNN + BiLSTM + CRF",
            ],
            "F1": [0.6846, 0.7163, 0.7413],
        }
    )

    save_barplot(
        architecture,
        x="Model",
        y="F1",
        title="Architecture Ablation - F1 Score",
        filename="architecture_ablation.png",
    )

    # 2. Ablazione embedding
    embeddings = pd.DataFrame(
        {
            "Embedding": [
                "FastText CC IT",
                "FastText Wiki IT",
                "GloVe 6B",
                "NLPL Word2Vec IT",
            ],
            "F1": [0.7265, 0.7413, 0.7501, 0.7521],
        }
    )

    save_barplot(
        embeddings,
        x="Embedding",
        y="F1",
        title="Embedding Ablation - F1 Score",
        filename="embedding_ablation.png",
    )

    # 3. Freeze vs fine-tuning
    freeze = pd.DataFrame(
        {
            "Setup": [
                "Fine-tuning",
                "Frozen embeddings",
            ],
            "F1": [0.7521, 0.7918],
        }
    )

    save_barplot(
        freeze,
        x="Setup",
        y="F1",
        title="Frozen vs Fine-tuned Embeddings - F1 Score",
        filename="freeze_vs_finetune.png",
    )

    # 4. Modelli finali
    final_models = pd.DataFrame(
        {
            "Model": [
                "WN-only",
                "All-datasets",
            ],
            "F1": [0.7918, 0.7774],
        }
    )

    save_barplot(
        final_models,
        x="Model",
        y="F1",
        title="Final Models - F1 Score",
        filename="final_models.png",
    )

    # 5. Generalista per dataset
    by_dataset = pd.DataFrame(
        {
            "Dataset": [
                "ADG",
                "FIC",
                "WN",
            ],
            "F1": [0.7323, 0.7758, 0.7836],
        }
    )

    save_barplot(
        by_dataset,
        x="Dataset",
        y="F1",
        title="Generalist Model Performance by Dataset",
        filename="generalist_by_dataset.png",
    )


if __name__ == "__main__":
    main()