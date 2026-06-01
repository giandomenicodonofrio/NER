#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

dataset_colors = {
    "ADG": "red",
    "FIC": "blue",
    "WN": "green",
}

def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_token(token: str) -> str:
    return token.strip()


def read_vec_subset(vec_path: Path, wanted_words: set[str]) -> dict[str, np.ndarray]:
    """
    Legge solo i vettori delle parole richieste.
    Supporta file .vec con o senza header.
    """
    wanted_exact = set(wanted_words)
    wanted_lower = {w.lower(): w for w in wanted_words}

    found: dict[str, np.ndarray] = {}

    with vec_path.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip().split()
        has_header = len(first) == 2 and first[0].isdigit() and first[1].isdigit()

        if not has_header:
            f.seek(0)

        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 3:
                continue

            word = parts[0]

            target_key = None
            if word in wanted_exact:
                target_key = word
            elif word.lower() in wanted_lower:
                target_key = wanted_lower[word.lower()]

            if target_key is None or target_key in found:
                continue

            try:
                vec = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue

            found[target_key] = vec

            if len(found) == len(wanted_words):
                break

    return found


def read_ner_dataset_words(
    dataset_dir: Path,
    top_k_entities: int,
    top_k_tokens: int,
) -> tuple[list[str], dict[str, str]]:
    """
    Estrae parole/entity frequenti da file BIO/TSV.
    - prende token più frequenti non-O
    - prende token più frequenti complessivi, filtrando punteggiatura e token cortissimi
    """
    entity_tokens = Counter()
    all_tokens = Counter()
    # token -> dataset
    word_sources: dict[str, str] = {}

    # token entity conteggiati per dataset
    entity_tokens_by_dataset: dict[str, Counter] = {}

    # token globali per dataset
    all_tokens_by_dataset: dict[str, Counter] = {}

    files = sorted(dataset_dir.rglob("*_train.tsv"))
    for path in files:
        dataset_name = path.parent.name
        entity_tokens_by_dataset.setdefault(dataset_name, Counter())
        all_tokens_by_dataset.setdefault(dataset_name, Counter())
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                token = parts[0].strip()
                label = parts[-1].strip()

                if not token or token.lower() in {"token", "word"}:
                    continue

                if is_useful_word(token):
                    all_tokens[token] += 1
                    all_tokens_by_dataset[dataset_name][token] += 1

                    if label != "O":
                        entity_tokens[token] += 1
                        entity_tokens_by_dataset[dataset_name][token] += 1

    words = []

    for dataset_name, counter in entity_tokens_by_dataset.items():
        for tok, _ in counter.most_common(top_k_entities):
            if tok not in words:
                words.append(tok)
                word_sources[tok] = dataset_name

    for dataset_name, counter in all_tokens_by_dataset.items():
        for tok, _ in counter.most_common(top_k_tokens):
            if tok not in words:
                words.append(tok)
                word_sources[tok] = dataset_name

    return words, word_sources


def is_useful_word(token: str) -> bool:
    if len(token) < 2:
        return False
    if re.fullmatch(r"\W+", token):
        return False
    if token.isdigit():
        return False
    return True


def reduce_vectors(vectors: np.ndarray, method: str, seed: int) -> np.ndarray:
    if len(vectors) < 2:
        raise ValueError("Servono almeno 2 parole trovate per visualizzare lo spazio.")

    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(vectors)

    if method == "tsne":
        perplexity = min(30, max(2, (len(vectors) - 1) // 3))
        return TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        ).fit_transform(vectors)

    raise ValueError(f"Metodo non supportato: {method}")


def plot_points(
    coords,
    words,
    title,
    out_path,
    max_labels,
    word_sources,
    dataset_colors,
):    
    plt.figure(figsize=(16, 12))
    colors = [
        dataset_colors.get(word_sources.get(word, ""), "gray")
        for word in words
    ]
    plt.scatter(coords[:, 0], coords[:, 1], s=28, alpha=0.75, c=colors, edgecolors="w", linewidths=0.5)

    # Etichetto un massimo di max_labels parole per evitare caos grafico.
    for i, word in enumerate(words[:max_labels]):
        plt.annotate(
            word,
            (coords[i, 0], coords[i, 1]),
            fontsize=7,
            alpha=0.85,
            xytext=(3, 3),
            textcoords="offset points",
        )

    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=dataset,
            markerfacecolor=color,
            markersize=8,
        )
        for dataset, color in dataset_colors.items()
    ]

    plt.legend(handles=legend_elements)
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_coords_csv(path: Path, model_name: str, method: str, words: list[str], coords: np.ndarray):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "method", "word", "x", "y"],
        )
        writer.writeheader()
        for word, xy in zip(words, coords):
            writer.writerow(
                {
                    "model": model_name,
                    "method": method,
                    "word": word,
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="embeddings/embeddings_manifest.json")
    parser.add_argument("--models", nargs="+", default=["fasttext_cc_it", "fasttext_wiki_it", "nlpl_it_word2vec", "glove_6b_300"])
    parser.add_argument("--words", nargs="*", default=None)
    parser.add_argument("--words-file", default=None)
    parser.add_argument("--dataset-dir", default="./dataset")
    parser.add_argument("--top-k-entities", type=int, default=150)
    parser.add_argument("--top-k-tokens", type=int, default=100)
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--out-dir", default="outputs/analysis/embedding_visualization")
    parser.add_argument("--max-labels", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = read_manifest(Path(args.manifest))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    method_dir = out_dir / args.method
    method_dir.mkdir(exist_ok=True)

    img_dir = method_dir / "images"
    img_dir.mkdir(exist_ok=True)
    data_dir = method_dir / "data"
    data_dir.mkdir(exist_ok=True)



    words: list[str] = []
    word_sources: dict[str, str] = {}

    if args.words:
        words.extend(args.words)

    if args.words_file:
        words.extend(
            [
                line.strip()
                for line in Path(args.words_file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )

    if args.dataset_dir:
        dataset_words, word_sources = read_ner_dataset_words(
            Path(args.dataset_dir),
            top_k_entities=args.top_k_entities,
            top_k_tokens=args.top_k_tokens,
        )

        words.extend(dataset_words)

    # Dedup mantenendo ordine.
    seen = set()
    words = [normalize_token(w) for w in words if not (normalize_token(w) in seen or seen.add(normalize_token(w)))]

    if not words:
        raise ValueError("Nessuna parola da visualizzare. Usa --words, --words-file oppure --dataset-dir.")

    coverage_rows = []

    for model_name in args.models:
        if model_name not in manifest:
            raise KeyError(f"Modello non trovato nel manifest: {model_name}")

        emb_path = Path(manifest[model_name]["path"])
        print(f"\n[{model_name}] Leggo subset vettori da {emb_path}")
        vectors_map = read_vec_subset(emb_path, set(words))

        found_words = [w for w in words if w in vectors_map]
        missing_words = [w for w in words if w not in vectors_map]

        print(f"[{model_name}] Trovate {len(found_words)}/{len(words)} parole")

        coverage_rows.append(
            {
                "model": model_name,
                "embedding_path": str(emb_path),
                "requested_words": len(words),
                "found_words": len(found_words),
                "missing_words": len(missing_words),
                "coverage": len(found_words) / len(words) if words else 0,
                "missing_sample": " ".join(missing_words[:30]),
            }
        )

        if len(found_words) < 2:
            print(f"[{model_name}] Skip: meno di 2 parole trovate.")
            continue

        matrix = np.vstack([vectors_map[w] for w in found_words])
        coords = reduce_vectors(matrix, method=args.method, seed=args.seed)

        png_path = img_dir / f"{model_name}_{args.method}.png"
        csv_path = data_dir / f"{model_name}_{args.method}_coords.csv"

        title = (
            f"{model_name} - {args.method.upper()} projection "
            f"({len(found_words)}/{len(words)} words)"
        )
        plot_points(
            coords=coords,
            words=found_words,
            title=title,
            out_path=png_path,
            max_labels=args.max_labels,
            word_sources=word_sources,
            dataset_colors=dataset_colors,
        )
        save_coords_csv(csv_path, model_name, args.method, found_words, coords)

        print(f"[{model_name}] PNG: {png_path}")
        print(f"[{model_name}] CSV: {csv_path}")

    coverage_path = out_dir / "coverage_summary.csv"
    with coverage_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "embedding_path",
                "requested_words",
                "found_words",
                "missing_words",
                "coverage",
                "missing_sample",
            ],
        )
        writer.writeheader()
        writer.writerows(coverage_rows)

    print(f"\nCoverage summary: {coverage_path}")


if __name__ == "__main__":
    main()
