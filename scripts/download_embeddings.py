#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


MODEL_REGISTRY = {
    "fasttext_cc_it": {
        "url": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.vec.gz",
        "family": "fastText",
        "algorithm": "CBOW + subword",
        "corpus": "Common Crawl + Wikipedia",
        "dim": 300,
        "format": "vec.gz",
        "license_note": "See FastText distribution terms.",
    },
    "fasttext_wiki_it": {
        "url": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.it.vec",
        "family": "fastText",
        "algorithm": "Skip-gram + subword",
        "corpus": "Wikipedia",
        "dim": 300,
        "format": "vec",
        "license_note": "See FastText distribution terms.",
    },
    "nlpl_it_word2vec": {
        "url": "https://vectors.nlpl.eu/repository/20/52.zip",
        "family": "Word2Vec",
        "algorithm": "Word2Vec Continuous Skipgram",
        "corpus": "Italian CoNLL17 corpus",
        "dim": 100,
        "format": "zip",
        "license_note": "See NLPL repository metadata inside the downloaded archive.",
    },
    "glove_6b_300": {
        "url": "https://nlp.stanford.edu/data/glove.6B.zip",
        "family": "GloVe",
        "algorithm": "Global Vectors / co-occurrence matrix factorization",
        "corpus": "Wikipedia 2014 + Gigaword 5",
        "dim": 300,
        "format": "zip",
        "license_note": "See Stanford GloVe distribution terms.",
    },
}


UNAVAILABLE_AUTOMATIC = {}


def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as response:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header else None
        downloaded = 0

        with out_path.open("wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    print(f"\r{out_path.name}: {downloaded / total * 100:6.2f}%", end="")
                else:
                    print(f"\r{out_path.name}: {downloaded / 1_000_000:,.1f} MB", end="")
    print()


def is_vec_like(path: Path) -> bool:
    return path.suffix.lower() in {".vec", ".txt", ".emb"}


def infer_dim_and_count(vec_path: Path) -> tuple[int | None, int]:
    dim = None
    count = 0

    with vec_path.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip().split()

        if len(first) == 2 and first[0].isdigit() and first[1].isdigit():
            dim = int(first[1])
        elif len(first) > 2:
            dim = len(first) - 1
            count += 1

        for line in f:
            if line.strip():
                count += 1

    return dim, count


def gunzip_to_vec(src: Path, dst: Path) -> None:
    with gzip.open(src, "rb") as f_in, dst.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def convert_word2vec_binary_to_text(src: Path, dst: Path) -> None:
    """
    Converte un file word2vec binary in text usando gensim.
    Richiede gensim installato.
    """
    try:
        from gensim.models import KeyedVectors
    except ImportError as exc:
        raise ImportError(
            "Per convertire file Word2Vec binari serve gensim. Installa con: pip install gensim"
        ) from exc

    kv = KeyedVectors.load_word2vec_format(str(src), binary=True, unicode_errors="ignore")
    kv.save_word2vec_format(str(dst), binary=False)


def extract_zip_to_vec(zip_path: Path, dst: Path) -> None:
    """
    Estrae da zip il primo file vettoriale utilizzabile.

    Ordine preferito:
    1. .vec/.txt/.emb
    2. .bin convertito con gensim
    """
    tmp_dir = zip_path.parent / f"{zip_path.stem}_extracted"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp_dir)

    files = [p for p in tmp_dir.rglob("*") if p.is_file()]

    text_candidates = [
        p for p in files
        if p.suffix.lower() in {".vec", ".txt", ".emb"}
        and "readme" not in p.name.lower()
        and "meta" not in p.name.lower()
    ]

    if text_candidates:
        selected = sorted(text_candidates, key=lambda p: p.stat().st_size, reverse=True)[0]
        print(f"Uso file testuale da zip: {selected.name}")
        shutil.copyfile(selected, dst)
        shutil.rmtree(tmp_dir)
        return

    bin_candidates = [
        p for p in files
        if p.suffix.lower() == ".bin"
        and "model" in p.name.lower()
    ]

    if not bin_candidates:
        bin_candidates = [p for p in files if p.suffix.lower() == ".bin"]

    if bin_candidates:
        selected = sorted(bin_candidates, key=lambda p: p.stat().st_size, reverse=True)[0]
        print(f"Converto binario Word2Vec da zip: {selected.name}")
        convert_word2vec_binary_to_text(selected, dst)
        shutil.rmtree(tmp_dir)
        return

    available = "\n".join(str(p.relative_to(tmp_dir)) for p in files[:50])
    shutil.rmtree(tmp_dir)
    raise ValueError(
        f"Impossibile trovare un file embedding utilizzabile in {zip_path}.\n"
        f"File trovati:\n{available}"
    )


def prepare_archive(raw_path: Path, final_vec_path: Path) -> None:
    name = raw_path.name.lower()

    if name.endswith(".vec") or name.endswith(".txt") or name.endswith(".emb"):
        shutil.copyfile(raw_path, final_vec_path)
    elif name.endswith(".gz"):
        gunzip_to_vec(raw_path, final_vec_path)
    elif name.endswith(".zip"):
        extract_zip_to_vec(raw_path, final_vec_path)
    elif name.endswith(".bin"):
        convert_word2vec_binary_to_text(raw_path, final_vec_path)
    else:
        raise ValueError(f"Formato non supportato: {raw_path}")


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def download_and_prepare(model_name: str, out_dir: Path, keep_raw: bool = False) -> None:
    if model_name not in MODEL_REGISTRY:
        if model_name in UNAVAILABLE_AUTOMATIC:
            raise ValueError(UNAVAILABLE_AUTOMATIC[model_name])
        raise KeyError(f"Modello sconosciuto: {model_name}")

    meta = MODEL_REGISTRY[model_name]
    url = meta["url"]

    raw_dir = out_dir / "raw"
    vec_dir = out_dir / "vec"
    raw_dir.mkdir(parents=True, exist_ok=True)
    vec_dir.mkdir(parents=True, exist_ok=True)

    raw_filename = Path(url.split("?")[0]).name
    raw_path = raw_dir / raw_filename
    final_vec_path = vec_dir / f"{model_name}.vec"

    if final_vec_path.exists():
        print(f"[SKIP] già pronto: {final_vec_path}")
    else:
        if not raw_path.exists():
            print(f"[DOWNLOAD] {model_name}")
            print(f"URL: {url}")
            download_file(url, raw_path)
        else:
            print(f"[SKIP] archivio già presente: {raw_path}")

        print(f"[PREPARE] {raw_path.name} -> {final_vec_path.name}")
        prepare_archive(raw_path, final_vec_path)

    dim, n_vectors = infer_dim_and_count(final_vec_path)

    manifest_path = out_dir / "embeddings_manifest.json"
    manifest = load_manifest(manifest_path)
    manifest[model_name] = {
        "name": model_name,
        "path": str(final_vec_path).replace("\\", "/"),
        "family": meta["family"],
        "algorithm": meta["algorithm"],
        "corpus": meta["corpus"],
        "dim": dim,
        "expected_dim": meta["dim"],
        "n_vectors": n_vectors,
        "source_url": url,
        "license_note": meta.get("license_note", ""),
    }
    save_manifest(manifest_path, manifest)

    if not keep_raw and raw_path.exists():
        raw_path.unlink()

    print(f"[OK] {model_name}: dim={dim}, vectors={n_vectors}, path={final_vec_path}")


def add_custom_url(name: str, url: str, family: str, out_dir: Path, keep_raw: bool = False) -> None:
    raw_dir = out_dir / "raw"
    vec_dir = out_dir / "vec"
    raw_dir.mkdir(parents=True, exist_ok=True)
    vec_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / Path(url.split("?")[0]).name
    final_vec_path = vec_dir / f"{name}.vec"

    if not final_vec_path.exists():
        if not raw_path.exists():
            print(f"[DOWNLOAD CUSTOM] {name}: {url}")
            download_file(url, raw_path)
        prepare_archive(raw_path, final_vec_path)

    dim, n_vectors = infer_dim_and_count(final_vec_path)
    manifest_path = out_dir / "embeddings_manifest.json"
    manifest = load_manifest(manifest_path)
    manifest[name] = {
        "name": name,
        "path": str(final_vec_path).replace("\\", "/"),
        "family": family,
        "algorithm": "custom",
        "corpus": "custom-url",
        "dim": dim,
        "expected_dim": dim,
        "n_vectors": n_vectors,
        "source_url": url,
    }
    save_manifest(manifest_path, manifest)

    if not keep_raw and raw_path.exists():
        raw_path.unlink()

    print(f"[OK] custom {name}: dim={dim}, vectors={n_vectors}, path={final_vec_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="embeddings")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=(
            "Modelli da scaricare. Usa 'all' per tutti quelli automatici. "
            f"Disponibili: {', '.join(MODEL_REGISTRY.keys())}"
        ),
    )
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--list", action="store_true", help="Mostra modelli disponibili ed esce")
    parser.add_argument("--custom-url", default=None, help="URL diretto custom a .vec/.txt/.gz/.zip/.bin")
    parser.add_argument("--name", default=None, help="Nome del modello custom")
    parser.add_argument("--family", default="custom", help="Famiglia del modello custom")

    args = parser.parse_args()

    if args.list:
        print("Modelli automatici disponibili:")
        for name, meta in MODEL_REGISTRY.items():
            print(f"- {name}: {meta['family']} | {meta['algorithm']} | {meta['corpus']} | {meta['dim']}d")
        print("\nNon automatici:")
        for name, reason in UNAVAILABLE_AUTOMATIC.items():
            print(f"- {name}: {reason}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.models == ["all"] or "all" in args.models:
        models = list(MODEL_REGISTRY.keys())
    else:
        models = args.models

    for model in models:
        download_and_prepare(model, out_dir=out_dir, keep_raw=args.keep_raw)

    if args.custom_url:
        if not args.name:
            raise ValueError("--name è obbligatorio con --custom-url")
        add_custom_url(
            name=args.name,
            url=args.custom_url,
            family=args.family,
            out_dir=out_dir,
            keep_raw=args.keep_raw,
        )

    print(f"\nManifest aggiornato: {out_dir / 'embeddings_manifest.json'}")


if __name__ == "__main__":
    main()
