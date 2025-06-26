import logging
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from surprise import SVD, BaselineOnly, Dataset, PredictionImpossible, Reader, accuracy
from surprise.model_selection import train_test_split

logger = logging.getLogger(__name__)


def prepare_surprise_dataset(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "movie_id",
    rating_col: str = "rating",
) -> Dataset:
    reader = Reader(rating_scale=(df[rating_col].min(), df[rating_col].max()))
    return Dataset.load_from_df(df[[user_col, item_col, rating_col]], reader)


def split_dataset(
    data: Dataset,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Any, Any]:
    trainset, testset = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
    )
    return trainset, testset


def train_collaborative_filtering_model(
    trainset: Any,
    n_factors: int = 100,
    n_epochs: int = 20,
    lr_all: float = 0.005,
    reg_all: float = 0.02,
) -> SVD:
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
    algo.fit(trainset)
    return algo


def train_baseline_model(
    trainset: Any,
) -> BaselineOnly:
    bsl_options = {
        "method": "sgd",
        "learning_rate": 0.005,
        "n_epochs": 10,
        "reg_u": 0.02,
        "reg_i": 0.02,
    }

    baseline = BaselineOnly(bsl_options=bsl_options)
    baseline.fit(trainset)
    return baseline


def evaluate_model(model: SVD | BaselineOnly, testset: Any) -> dict[str, float]:
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    return {"rmse": rmse, "mae": mae}


def get_predictions(
    model: SVD | BaselineOnly,
    testset: Any,
) -> pd.DataFrame:
    records = []

    for uid, iid, true_r in testset:
        try:
            pred = model.predict(uid, iid)
            est = pred.est
        except PredictionImpossible:
            est = None

        records.append(
            {
                "user_id": uid,
                "movie_id": iid,
                "actual": true_r,
                "predicted": est,
                "estimator_name": type(model).__name__,
            }
        )

    return pd.DataFrame(records)


def plot_error_distribution(preds: pd.DataFrame) -> None:
    df = preds.dropna(subset=["predicted"]).copy()
    df["abs_error"] = (df["predicted"] - df["actual"]).abs()

    plt.figure()
    df["abs_error"].hist(bins=20)

    plt.xlabel("Absolute Error")
    plt.ylabel("Count")
    plt.title("Distribution of Absolute Errors")

    plt.show()


def train_and_evaluate_models(df: pd.DataFrame) -> None:
    data = prepare_surprise_dataset(df)
    trainset, testset = split_dataset(data)

    cf_model = train_collaborative_filtering_model(trainset)
    cf_metrics = evaluate_model(cf_model, testset)
    logger.info(
        f"Collaborative Filtering metrics: RMSE = {cf_metrics['rmse']:.4f}, MAE = {cf_metrics['mae']:.4f}"
    )

    baseline_model = train_baseline_model(trainset)
    baseline_metrics = evaluate_model(baseline_model, testset)
    logger.info(
        f"Baseline metrics: RMSE = {baseline_metrics['rmse']:.4f}, MAE = {baseline_metrics['mae']:.4f}"
    )

    svd_preds = get_predictions(cf_model, testset)
    plot_error_distribution(svd_preds)
