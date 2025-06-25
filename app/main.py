from app.constants import DATASET_MOVIES, DATASET_RATINGS
from app.preprocessing import ingest_and_preprocess_data
from app.utils import save_dataframe


def main() -> None:
    data_set_file_paths = (DATASET_RATINGS, DATASET_MOVIES)
    df_preprocessed = ingest_and_preprocess_data(data_set_file_paths)
    save_dataframe(df_preprocessed, "preprocessed_data.csv")


if __name__ == "__main__":
    main()
