"""Palmer Penguins dataset loading and splitting for Project 2.

The dataset is vendored as ``data/penguins.csv`` (canonical palmerpenguins data).
Rows with missing values are dropped, which leaves ~333 complete penguins.
"""

import os

import numpy as np
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "penguins.csv")

TARGET = "species"
NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
CATEGORICAL_FEATURES = ["island", "sex", "year"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Fixed for reproducibility across requests.
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_penguins():
    """Return the cleaned penguins DataFrame (features + target, no missing values)."""
    df = pd.read_csv(CSV_PATH)
    df = df[FEATURES + [TARGET]].dropna().reset_index(drop=True)
    # Treat ``year`` as a discrete category rather than a number.
    df["year"] = df["year"].astype(int).astype(str)
    df["island"] = df["island"].astype(str)
    df["sex"] = df["sex"].astype(str)
    return df


def species_classes(df):
    """Sorted list of the three species labels."""
    return sorted(df[TARGET].unique().tolist())


def get_split(df):
    """Stratified 80/20 train/test split with a fixed seed."""
    from sklearn.model_selection import train_test_split

    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def mad(values):
    """Median absolute deviation, with a positive fallback for constant features."""
    values = np.asarray(values, dtype=float)
    m = np.median(np.abs(values - np.median(values)))
    if m == 0:
        m = np.std(values)
    return m if m > 0 else 1.0


def numeric_mads(df):
    """MAD for each numeric feature, used to weight the L1 counterfactual distance."""
    return {col: mad(df[col].values) for col in NUMERIC_FEATURES}
