from pathlib import Path

import pandas as pd

output_dir = Path("outputs/analysis")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    "outputs/final/final_all_datasets/predictions/test_errors.tsv",
    sep="\t",
)

confusions = (
    df.groupby(["gold", "pred"])
      .size()
      .reset_index(name="count")
      .sort_values("count", ascending=False)
)

confusions.to_csv(
    output_dir / "top_confusions.csv",
    index=False,
)

tokens = (
    df.groupby(["token", "gold", "pred"])
      .size()
      .reset_index(name="count")
      .sort_values("count", ascending=False)
)

tokens.to_csv(
    output_dir / "top_error_tokens.csv",
    index=False,
)