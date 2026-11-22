from dotenv import load_dotenv
from os import getenv
from src.processing import DataProcessor


def main() -> None:
    load_dotenv()
    dp = DataProcessor(
        *list(
            map(getenv, ["DB_USER", "DB_PWD", "DB_HOST", "DB_PORT", "DB_NAME"])
        )
    )
    df = dp.preprocessed()
    user_df = dp.user_profile(df)
    dp.retention_event_type(user_df)
    dp.retention_device(user_df)
    dp.retention_region(user_df)
    dp.first_hypothesis(user_df)
    dp.second_hypothesis(user_df)
    dp.cor_heatmap(dp.correlation_matrix(user_df))


if __name__ == "__main__":
    main()
