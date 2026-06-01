"""Command-line entry point for one configured training or evaluation run."""

import argparse
import json
from pathlib import Path

import torch

from src.utils.config import load_experiment_config
from src.training.seed import set_seed
from src.data.datamodule import build_datasets_and_vocabs, build_dataloaders
from src.preprocessing.embeddings import build_embedding_matrix
from src.models.sequence_tagger import SequenceTagger
from src.training.trainer import Trainer


def save_vocabs(vocabs: dict, output_dir: Path) -> None:
    """Persist vocabularies so model outputs can be inspected after training."""
    vocab_dir = output_dir / "vocabs"
    vocab_dir.mkdir(parents=True, exist_ok=True)

    for name, vocab in vocabs.items():
        path = vocab_dir / f"{name}_vocab.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "itos": vocab.itos,
                    "stoi": vocab.stoi,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )


def save_config(config: dict, output_dir: Path) -> None:
    """Persist the fully merged config used by the current experiment."""
    path = output_dir / "config_resolved.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path alla config esperimento YAML",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda, cpu oppure lasciare vuoto per autodetect",
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Carica il checkpoint best.pt e valuta senza rifare training",
    )

    args = parser.parse_args()

    config = load_experiment_config(args.config)

    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, output_dir)

    print(f"Experiment: {config['experiment']['name']}")
    print(f"Output dir: {output_dir}")

    print("Loading datasets...")
    datasets, vocabs = build_datasets_and_vocabs(config)

    print(f"Train sentences: {len(datasets['train'])}")
    print(f"Dev sentences:   {len(datasets['dev'])}")
    print(f"Test sentences:  {len(datasets['test'])}")

    print(f"Token vocab:   {len(vocabs['token'])}")
    print(f"Char vocab:    {len(vocabs['char'])}")
    print(f"Label vocab:   {len(vocabs['label'])}")
    print(f"Dataset vocab: {len(vocabs['dataset'])}")
    print(f"Labels: {vocabs['label'].itos}")

    save_vocabs(vocabs, output_dir)

    print("Building dataloaders...")
    dataloaders = build_dataloaders(config, datasets)

    print("Loading embedding matrix...")
    embedding_matrix = build_embedding_matrix(
        token_vocab=vocabs["token"],
        embedding_config=config["embedding"],
        preprocessing_config=config["preprocessing"],
    )

    print("Building model...")
    model = SequenceTagger(
        embedding_matrix=embedding_matrix,
        num_chars=len(vocabs["char"]),
        num_labels=len(vocabs["label"]),
        config=config,
        pad_token_id=vocabs["token"].stoi["<PAD>"],
        pad_char_id=vocabs["char"].stoi["<PAD>"],
    )

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        config=config,
        label_vocab=vocabs["label"],
        device=args.device,
    )

    print("Starting training...")
    
    if args.eval_only:
        print("Eval-only mode: loading best checkpoint...")
    else:
        print("Starting training...")
        trainer.train()

    print("Loading best checkpoint...")
    best_path = output_dir / "checkpoints" / "best.pt"
    checkpoint = torch.load(
        best_path,
        map_location=trainer.device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Evaluating on test...")
    test_metrics = trainer.evaluate(split="test", save_outputs=True)

    metrics_path = output_dir / "metrics" / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    print("Test metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    print(f"Saved test metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
