"""AG News dataset download, loading and splitting for Project 3.

The dataset (https://huggingface.co/datasets/fancyzhx/ag_news) is downloaded once
as parquet into ``data/`` (gitignored, ~30 MB). Columns: ``text``, ``label`` with
0=World, 1=Sports, 2=Business, 3=Sci/Tech.

Split protocol: the official 120k train set is split 110k/10k into train/validation
(fixed seed). All tuning (deferral threshold, active-learning monitoring) uses the
validation split; the official 7.6k test set is used only for final evaluation.
"""

import os
import urllib.request

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
TEST_PATH = os.path.join(DATA_DIR, "test.parquet")

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
N_CLASSES = 4

RANDOM_STATE = 42
VAL_SIZE = 10_000

_BASE = "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data"
_URLS = {
    TRAIN_PATH: [
        f"{_BASE}/train-00000-of-00001.parquet",
        "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    ],
    TEST_PATH: [
        f"{_BASE}/test-00000-of-00001.parquet",
        "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
    ],
}
_EXPECTED_ROWS = {TRAIN_PATH: 120_000, TEST_PATH: 7_600}


def dataset_available():
    return os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH)


def download(force=False):
    """Fetch the two official parquet shards and verify their row counts."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for path, urls in _URLS.items():
        if os.path.exists(path) and not force:
            print(f"already present: {path}")
            continue
        last_err = None
        for url in urls:
            try:
                print(f"downloading {url} ...")
                urllib.request.urlretrieve(url, path)
                break
            except Exception as err:  # noqa: BLE001 - try the fallback URL
                last_err = err
        else:
            raise RuntimeError(
                f"Could not download {path}. Tried: {urls}. Last error: {last_err}. "
                "Download the AG News parquet files manually from "
                "https://huggingface.co/datasets/fancyzhx/ag_news and place them "
                f"in {DATA_DIR} as train.parquet / test.parquet."
            )
        n = len(pd.read_parquet(path))
        expected = _EXPECTED_ROWS[path]
        if n != expected:
            raise RuntimeError(f"{path}: expected {expected} rows, got {n}")
        print(f"ok: {path} ({n} rows)")


def _load(path):
    df = pd.read_parquet(path)[["text", "label"]]
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def load_train_val():
    """Stratified 110k/10k train/validation split of the official train set.

    Both frames keep a ``key`` column: the row's position in the official train
    parquet. Experts derive their per-example randomness from these keys, so
    expert answers are stable across tasks regardless of the split.
    """
    from sklearn.model_selection import train_test_split

    df = _load(TRAIN_PATH)
    df["key"] = df.index
    train_df, val_df = train_test_split(
        df, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_test():
    """Official test set, with keys offset so they never collide with train keys."""
    df = _load(TEST_PATH)
    df["key"] = df.index + 1_000_000
    return df
