import logging

from app.constants import DATASET_MOVIES, DATASET_RATINGS
from app.eda import run_eda
from app.preprocessing import ingest_and_preprocess_data
from app.train import train_and_evaluate_models
from app.utils import save_dataframe

logging.basicConfig(level=logging.INFO)


def main() -> None:
    data_set_file_paths = (DATASET_RATINGS, DATASET_MOVIES)
    df_preprocessed = ingest_and_preprocess_data(data_set_file_paths)
    save_dataframe(df_preprocessed, "preprocessed_data.csv")

    run_eda(df_preprocessed)
    train_and_evaluate_models(df_preprocessed)


if __name__ == "__main__":
    main()
