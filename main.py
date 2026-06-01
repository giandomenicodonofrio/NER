"""Run one or more experiment configs in isolated Python subprocesses.

Use ``--only`` to select configs from the command line and ``--start-from`` to
resume a predefined sequence after an interrupted run.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# EXPERIMENTS = [ # experiments for ablation of preprocessing
#     "configs/experiment/all_datasets/p0_base.yaml",
#     "configs/experiment/all_datasets/p1_normalized_embeddings.yaml",
#     "configs/experiment/all_datasets/p2_no_stopwords_normalized_embeddings.yaml",
#     "configs/experiment/wn_only/p0_base.yaml",
#     "configs/experiment/wn_only/p1_normalized_embeddings.yaml",
#     "configs/experiment/wn_only/p2_no_stopwords_normalized_embeddings.yaml",
# ]

# EXPERIMENTS = [  # experiments for ablation of architecture components
#     "configs/experiment/architecture/wn_a0_bilstm_softmax.yaml",
#     "configs/experiment/architecture/wn_a1_bilstm_crf.yaml",
#     "configs/experiment/architecture/wn_a2_charcnn_bilstm_crf.yaml",
# ]

# EXPERIMENTS = [ # experiments with different embeddings
#     "configs/experiment/embeddings/wn_fasttext_wiki_it.yaml",
#     "configs/experiment/embeddings/wn_fasttext_cc_it.yaml",
#     "configs/experiment/embeddings/wn_nlpl_it_word2vec.yaml",
#     "configs/experiment/embeddings/wn_glove_6b_300.yaml",
# ]

# EXPERIMENTS = [ # experiments with different embeddings + freeze/fine-tune
#     "configs/experiment/freeze/wn_nlpl_finetune.yaml",
#     "configs/experiment/freeze/wn_nlpl_frozen.yaml",
# ]

# EXPERIMENTS = [ # final experiments for error analysis
#     "configs/experiment/final/final_all_datasets.yaml",
#     "configs/experiment/final/final_wn_only.yaml",
# ]

# EXPERIMENTS = [ # experiments for comparison of random vs balanced sampling
#     "configs/experiment/balancing/all_balanced_sampling.yaml",
# ]


# EXPERIMENTS = [ # experiments for tuning hyperparameters
#     "configs/experiment/tuning/tuning_wn_lr_0005.yaml",
#     "configs/experiment/tuning/tuning_wn_word_dropout_010.yaml",
#     "configs/experiment/tuning/tuning_wn_charcnn_100.yaml",
#     "configs/experiment/tuning/tuning_wn_entity_weighted.yaml",
# ]

# Default sequence. Prefer ``--only`` for ad-hoc runs; all available configs are
# grouped by experimental phase under ``configs/experiment``.
EXPERIMENTS = [
    "configs/experiment/post_tuning/post_tuning_all_datasets_word_dropout_010.yaml",
]



def run_experiment(config_path: str, device: str | None = None, eval_only: bool = False) -> int:
    """Run one experiment without leaking process state into the next run."""
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
