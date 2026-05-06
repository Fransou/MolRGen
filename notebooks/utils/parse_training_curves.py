"""
Utility function to parse training curve column names and reshape dataframes.
"""

import re
from typing import Tuple

import pandas as pd


def parse_column_name(col_name: str) -> Tuple[int, str]:
    group_match = re.search(r"group_(\d+)", col_name)
    group_size = int(group_match.group(1)) if group_match else 8

    metric = col_name.split("-")[-1].replace(" ", "")

    return group_size, metric


def load_training_curves(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[c for c in df.columns if re.match("Group: 260322_.*ms_4", c) is not None]]
    # Change df columns to multi-indexed
    multi_index = pd.MultiIndex.from_tuples(
        [parse_column_name(col) for col in df.columns], names=["group_size", "metric"]
    )
    df.columns = multi_index

    # Unpivot the dataframe
    df = df.reset_index(names=["step"])
    df = df.melt(
        id_vars=[c for c in df.columns if c not in multi_index],
        var_name=["group_size", "metric"],
        value_name="value",
    )
    df.rename(columns={("step", ""): "step"}, inplace=True)
    return df
