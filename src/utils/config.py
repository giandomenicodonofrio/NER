"""Load and compose the YAML files that define an experiment."""

from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> dict:
    """Read one YAML file and fail early when its path is invalid."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict, update: dict) -> dict:
    """Recursively merge ``update`` into ``base`` and return the mutated base."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_experiment_config(path: str | Path) -> dict:
    """Resolve an experiment config into the dictionary consumed by training.

    Experiments are intentionally split into reusable model, embedding and
    preprocessing YAML files. They are merged first; values declared directly
    in the experiment file are applied last and therefore act as overrides.
    """
    config = load_yaml(path)

    merged = {}

    for key in ["model_config", "embedding_config", "preprocessing_config"]:
        if key in config:
            partial = load_yaml(config[key])
            merged = deep_update(merged, partial)

    merged = deep_update(merged, config)

    embedding_dim = merged.get("embedding", {}).get("dim")
    if embedding_dim is not None and "model" in merged:
        # The matrix loaded at runtime is authoritative when configs are mixed.
        merged["model"]["word_embedding_dim"] = embedding_dim

    return merged
