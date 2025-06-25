import contextlib
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_csv_data(
    file_paths: list[Path],
) -> dict[str, pd.DataFrame]:
    return {path.stem: pd.read_csv(path) for path in file_paths}


def merge_datasets(
    datasets: dict[str, pd.DataFrame],
    on: list[str],
) -> pd.DataFrame:
    dfs = list(datasets.values())
    merged_df = dfs[0].copy()

    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on=on, how="inner")

    return merged_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic cleaning operations."""
    df = df.drop_duplicates()

    for col in df.columns:
        if "timestamp" in col.lower():
            with contextlib.suppress(ValueError, TypeError):
                df[col] = pd.to_datetime(df[col], unit="s")

    missing_share_threshold = 0.1
    columns_with_high_missing = df.columns[
        df.isnull().mean() >= missing_share_threshold
    ].tolist()
    if columns_with_high_missing:
        raise ValueError(
            f"Columns with much missing data (threshold={missing_share_threshold}):"
            f"{columns_with_high_missing}"
        )

    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["release_year"] = (
        df["title"].str.extract(r"\((\d{4})\)$", expand=False).astype("Int64")
    )

    df["clean_title"] = df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)

    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["rating_year"] = df["timestamp"].dt.year
        df["rating_month"] = df["timestamp"].dt.month
        df["rating_day"] = df["timestamp"].dt.day
        df["rating_hour"] = df["timestamp"].dt.hour
    else:
        logger.warning(
            "Column 'timestamp' is not datetime, skipping datetime features."
        )

    return df


def normalize_ratings(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()
    ratings = df["rating"].astype(float)

    min_r, max_r = ratings.min(), ratings.max()
    df["rating_scaled"] = (ratings - min_r) / (max_r - min_r)
    df["rating_zscore"] = (ratings - ratings.mean()) / ratings.std()

    return df


def ingest_and_preprocess_data(
    file_paths: tuple[Path, Path],
) -> None:
    datasets = load_csv_data(file_paths)

    df_merged = merge_datasets(datasets, on=["item_id"])
    df_clean = clean_data(df_merged)

    df_features = extract_features(df_clean)

    return normalize_ratings(df_features)
