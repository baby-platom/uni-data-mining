import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def data_overview(df: pd.DataFrame) -> tuple[int, int, int]:
    """Compute total number of ratings, unique users, and unique movies."""
    total_ratings = len(df)
    num_users = df["user_id"].nunique()
    num_movies = df["movie_id"].nunique()

    logger.info(f"Total ratings: {total_ratings}")
    logger.info(f"Unique users: {num_users}")
    logger.info(f"Unique movies: {num_movies}")

    return total_ratings, num_users, num_movies


def compute_matrix_sparsity(df: pd.DataFrame, num_users: int, num_movies: int) -> float:
    """Compute fraction of observed (user, movie) rating pairs."""
    possible_pairs = num_users * num_movies
    observed = len(df.drop_duplicates(subset=["user_id", "movie_id"]))
    sparsity = observed / possible_pairs

    logger.info(f"Matrix sparsity (fraction observed): {sparsity:.6f}")

    return sparsity


def plot_raw_rating_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    ratings = df["rating"].dropna().astype(int)
    min_r, max_r = ratings.min(), ratings.max()
    bins = np.arange(min_r - 0.5, max_r + 1.5, 1)

    ax.hist(ratings, bins=bins, edgecolor="black")

    ax.set_xticks(np.arange(min_r, max_r + 1))
    ax.set_title("Raw Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frequency")


def user_activity_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    counts = df.groupby("user_id").size()
    counts.hist(bins=50, ax=ax)

    ax.set_title("User Activity Distribution")
    ax.set_xlabel("Number of Ratings per User")
    ax.set_ylabel("Count of Users")


def movie_popularity_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Count ratings per movie and plot distribution."""
    counts = df.groupby("movie_id").size()
    counts.hist(bins=50, ax=ax)

    ax.set_title("Movie Popularity Distribution")
    ax.set_xlabel("Number of Ratings per Movie")
    ax.set_ylabel("Count of Movies")


def release_year_spread(df: pd.DataFrame, ax: plt.Axes) -> None:
    df["release_year"].hist(bins=30, ax=ax)

    ax.set_title("Release Year Spread")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Movies")


def inspect_missing_values(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()

    logger.info("Missing values per column:")
    logger.info(missing)


def compute_feature_correlations(df: pd.DataFrame, features: list[str]) -> None:
    if missing := [f for f in features if f not in df.columns]:
        raise ValueError(f"Features not found in DataFrame: {missing}")
    numeric_df = df[features].select_dtypes(include=[np.number])

    corr = numeric_df.corr()

    _, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr)
    plt.colorbar(cax, ax=ax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    ax.set_title("Feature Correlation Matrix")

    plt.show()


def detect_outliers(df: pd.DataFrame) -> dict[str, list[int]]:
    """Identify users and movies with extreme average ratings."""
    results: dict[str, list[int]] = {"users": [], "movies": []}

    grouping = df.groupby("user_id")["rating_zscore"]
    user_means = grouping.mean()

    threshold = 3
    outlier_users = user_means[abs(user_means) > threshold].index.tolist()
    results["users"] = outlier_users

    logger.info(f"Users with extreme z-score means (>3 sigma): {outlier_users}")

    grouping_m = df.groupby("movie_id")["rating_zscore"]
    movie_means = grouping_m.mean()

    threshold = 3
    outlier_movies = movie_means[abs(movie_means) > 3].index.tolist()
    results["movies"] = outlier_movies

    logger.info(f"Movies with extreme z-score means (>3 sigma): {outlier_movies}")

    return results


def run_eda(df: pd.DataFrame) -> None:
    _, num_users, num_movies = data_overview(df)
    compute_matrix_sparsity(df, num_users, num_movies)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_raw_rating_distribution(df, ax=axes[0, 0])
    user_activity_distribution(df, ax=axes[0, 1])
    movie_popularity_distribution(df, ax=axes[1, 0])
    release_year_spread(df, ax=axes[1, 1])
    fig.tight_layout()
    plt.show()

    inspect_missing_values(df)
    compute_feature_correlations(
        df,
        features=[
            "user_id",
            "movie_id",
            "rating",
            "release_year",
            "rating_year",
            "rating_month",
            "rating_day",
            "rating_hour",
        ],
    )
    detect_outliers(df)
