import argparse
import os
import subprocess
import sys
from pathlib import Path


# EXPERIMENTS = [ # esperimenti per ablazione preprocessing
#     "configs/experiment/all_datasets/p0_base.yaml",
#     "configs/experiment/all_datasets/p1_normalized_embeddings.yaml",
#     "configs/experiment/all_datasets/p2_no_stopwords_normalized_embeddings.yaml",
#     "configs/experiment/wn_only/p0_base.yaml",
#     "configs/experiment/wn_only/p1_normalized_embeddings.yaml",
#     "configs/experiment/wn_only/p2_no_stopwords_normalized_embeddings.yaml",
# ]

# EXPERIMENTS = [  # esperimenti per ablazione architettura
#     "configs/experiment/architecture/wn_a0_bilstm_softmax.yaml",
#     "configs/experiment/architecture/wn_a1_bilstm_crf.yaml",
#     "configs/experiment/architecture/wn_a2_charcnn_bilstm_crf.yaml",
# ]

# EXPERIMENTS = [ # esperimenti con embeddings diversi
#     "configs/experiment/embeddings/wn_fasttext_wiki_it.yaml",
#     "configs/experiment/embeddings/wn_fasttext_cc_it.yaml",
#     "configs/experiment/embeddings/wn_nlpl_it_word2vec.yaml",
#     "configs/experiment/embeddings/wn_glove_6b_300.yaml",
# ]

# EXPERIMENTS = [ # esperimenti con embeddings diversi + freeze/fine-tune
#     "configs/experiment/freeze/wn_nlpl_finetune.yaml",
#     "configs/experiment/freeze/wn_nlpl_frozen.yaml",
# ]

# EXPERIMENTS = [ # esperimenti finali per analisi errori
#     "configs/experiment/final/final_all_datasets.yaml",
#     "configs/experiment/final/final_wn_only.yaml",
# ]

# EXPERIMENTS = [ # esperimenti per confronto random vs balanced sampling
#     "configs/experiment/balancing/all_balanced_sampling.yaml",
# ]


# EXPERIMENTS = [ # esperimenti per tuning iperparametri
#     "configs/experiment/tuning/tuning_wn_lr_0005.yaml",
#     "configs/experiment/tuning/tuning_wn_word_dropout_010.yaml",
#     "configs/experiment/tuning/tuning_wn_charcnn_100.yaml",
#     "configs/experiment/tuning/tuning_wn_entity_weighted.yaml",
# ]

EXPERIMENTS = [
    "configs/experiment/post_tuning/post_tuning_all_datasets_word_dropout_010.yaml",
]



def run_experiment(config_path: str, device: str | None = None, eval_only: bool = False) -> int:
    command = [
        sys.executable,
        "-m",
        "src.scripts.train",
        "--config",
        config_path,
    ]

    if device is not None:
        command.extend(["--device", device])

    if eval_only:
        command.append("--eval-only")

    env = os.environ.copy()

    project_root = Path(__file__).resolve().parents[3]
    src_path = project_root / "src"

    env["PYTHONPATH"] = str(src_path)

    print("\n" + "=" * 80)
    print(f"Running experiment: {config_path}")
    print("=" * 80)

    result = subprocess.run(command, env=env)

    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--start-from", type=str, default=None)
    parser.add_argument("--only", type=str, nargs="*", default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
    )

    args = parser.parse_args()

    experiments = EXPERIMENTS

    if args.only:
        experiments = args.only

    if args.start_from:
        if args.start_from not in experiments:
            raise ValueError(f"{args.start_from} non è nella lista esperimenti")

        experiments = experiments[experiments.index(args.start_from):]

    for config_path in experiments:
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config non trovata: {config_path}")

        return_code = run_experiment(config_path, device=args.device, eval_only=args.eval_only)

        if return_code != 0:
            print(f"Experiment failed: {config_path}")
            sys.exit(return_code)

    print("\nTutti gli esperimenti completati.")


if __name__ == "__main__":
    main()




















# from src.utils.config import load_experiment_config
# from src.utils.reader import load_datasets_split
# from src.utils.vocabulary import build_token_vocab, build_char_vocab, build_label_vocab

# config = load_experiment_config("configs/experiment/all_datasets/p0_base.yaml")

# train = load_datasets_split(config, "train")

# token_vocab = build_token_vocab(train)
# char_vocab = build_char_vocab(train)
# label_vocab = build_label_vocab(train)

# print(len(train))
# print(len(token_vocab), len(char_vocab), len(label_vocab))
# print(label_vocab.itos)

# -------------------------------------------------------------------

# from src.utils.config import load_experiment_config
# from src.data.datamodule import build_datasets_and_vocabs, build_dataloaders

# config = load_experiment_config("configs/experiment/all_datasets/p0_base.yaml")

# datasets, vocabs = build_datasets_and_vocabs(config)
# loaders = build_dataloaders(config, datasets)

# batch = next(iter(loaders["train"]))

# print(batch["token_ids"].shape)
# print(batch["char_ids"].shape)
# print(batch["label_ids"].shape)
# print(batch["mask"].shape)
# print(vocabs["label"].itos)

# -------------------------------------------------------------------

# from src.utils.config import load_experiment_config
# from src.data.datamodule import build_datasets_and_vocabs
# from src.preprocessing.embeddings import build_embedding_matrix

# config = load_experiment_config("configs/experiment/all_datasets/p1_normalized_embeddings.yaml")

# datasets, vocabs = build_datasets_and_vocabs(config)

# embedding_matrix = build_embedding_matrix(
#     token_vocab=vocabs["token"],
#     embedding_config=config["embedding"],
#     preprocessing_config=config["preprocessing"],
# )

# print(embedding_matrix.shape)
# print(embedding_matrix.dtype)

# -------------------------------------------------------------------

# from src.utils.config import load_experiment_config
# from src.data.datamodule import build_datasets_and_vocabs
# from src.preprocessing.embeddings import build_embedding_matrix

# config = load_experiment_config("configs/experiment/all_datasets/p1_normalized_embeddings.yaml")

# datasets, vocabs = build_datasets_and_vocabs(config)

# embedding_matrix = build_embedding_matrix(
#     token_vocab=vocabs["token"],
#     embedding_config=config["embedding"],
#     preprocessing_config=config["preprocessing"],
# )

# print(embedding_matrix.shape)
# print(embedding_matrix.dtype)


# -------------------------

# from src.utils.config import load_experiment_config
# from src.data.datamodule import build_datasets_and_vocabs, build_dataloaders
# from src.preprocessing.embeddings import build_embedding_matrix
# from src.models.charcnn_bilstm_crf import CharCNNBiLSTMCRF

# config = load_experiment_config("configs/experiment/all_datasets/p0_base.yaml")

# datasets, vocabs = build_datasets_and_vocabs(config)
# loaders = build_dataloaders(config, datasets)

# embedding_matrix = build_embedding_matrix(
#     token_vocab=vocabs["token"],
#     embedding_config=config["embedding"],
#     preprocessing_config=config["preprocessing"],
# )

# model = CharCNNBiLSTMCRF(
#     embedding_matrix=embedding_matrix,
#     num_chars=len(vocabs["char"]),
#     num_labels=len(vocabs["label"]),
#     config=config,
# )

# batch = next(iter(loaders["train"]))

# loss = model(
#     token_ids=batch["token_ids"],
#     char_ids=batch["char_ids"],
#     mask=batch["mask"],
#     labels=batch["label_ids"],
# )

# print(loss)

# preds = model.decode(
#     token_ids=batch["token_ids"],
#     char_ids=batch["char_ids"],
#     mask=batch["mask"],
# )

# print(len(preds))
# print(preds[0])