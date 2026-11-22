from dataclasses import dataclass
from typing import ClassVar
from sqlalchemy import create_engine
from matplotlib import pyplot as plt
from phik import phik_matrix
import pandas as pd
import seaborn as sns


def pretty_describe(df: pd.DataFrame) -> pd.DataFrame:
    return round(df.describe(), 2)


@dataclass(frozen=True)
class DataProcessor:
    user: str
    pwd: str
    host: str
    port: str
    name: str
    connection_string: ClassVar[str] = "postgresql://{}:{}@{}:{}/{}"
    rate_path: ClassVar[str] = "data/final_tickets_tenge_df.csv"

    @staticmethod
    def to_rub(row):
        if row["currency_code"] == "rub":
            return row["revenue"]
        return row["revenue"] * row["curs"] / row["nominal"]

    def read_db(self) -> pd.DataFrame:
        connection_string = self.connection_string.format(
            self.user, self.pwd, self.host, self.port, self.name
        )
        engine = create_engine(connection_string)
        with open("src/sql/df.sql", "r") as file:
            query = file.read()
        df = pd.read_sql_query(query, con=engine)
        return df

    def read_rate(self) -> pd.DataFrame:
        df = pd.read_csv(self.rate_path)
        return df

    def preprocessed(self) -> pd.DataFrame:
        df = self.read_db()
        tickets_tenge = self.read_rate()
        tickets_tenge["data"] = pd.to_datetime(tickets_tenge["data"])

        mdf = df.merge(tickets_tenge, left_on="order_dt", right_on="data")
        mdf["revenue_rub"] = mdf.apply(self.to_rub, axis=1)
        mdf = mdf[[*df.columns, "revenue_rub"]]
        df = mdf
        df["order_id"] = df["order_id"].astype(str)
        df["event_id"] = df["event_id"].astype(str)
        df = df[
            (df["revenue_rub"] <= df["revenue_rub"].quantile(0.99))
            & (df["revenue_rub"] > 0)
        ]
        df = df.drop_duplicates(
            subset=[
                "user_id",
                "tickets_count",
                "event_id",
                "order_dt",
                "order_ts",
                "device_type_canonical",
                "service_name",
                "revenue_rub",
            ]
        )
        print(f"Осталось {len(df)} записей.")
        return df

    def user_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        user_df = (
            df.groupby("user_id")
            .agg(
                {
                    "order_dt": "min",
                    "order_ts": "min",
                    "event_id": "count",
                    "revenue_rub": "mean",
                    "days_since_prev": "mean",
                }
            )
            .reset_index()
        )
        user_df = user_df.merge(df, on=["user_id", "order_dt", "order_ts"])[
            [
                "user_id",
                "order_dt",
                "device_type_canonical",
                "region_name",
                "event_type_main",
                "event_id_x",
                "days_since_prev_x",
                "revenue_rub_x",
            ]
        ]
        user_df.columns = [
            "user_id",
            "min_order_dt",
            "device_type_canonical",
            "region_name",
            "event_type_main",
            "event_count",
            "days_since_prev",
            "mean_revenue_rub",
        ]
        helper_grouped = (
            df.groupby("user_id").agg({"order_dt": "max"}).reset_index()
        )
        helper_grouped.columns = ["user_id", "max_order_dt"]
        user_df = user_df.merge(helper_grouped, on="user_id")
        user_df["is_two"] = user_df.apply(
            lambda row: 1 if row["event_count"] >= 2 else 0, axis=1
        )
        user_df = user_df[
            user_df["event_count"] <= user_df["event_count"].quantile(0.99)
        ]
        return user_df

    @staticmethod
    def _retention(grouped: pd.DataFrame) -> None:
        print(pretty_describe(grouped))
        grouped.plot(kind="bar", rot=0, figsize=(10, 6))
        plt.legend(["До 2", "От 2"])
        plt.ylabel("Количество пользователей")
        plt.xlabel("Тип мероприятия")
        plt.title(
            "Отношение клиентов, которые совершили 2 и более заказа к тем, кто совершил менее"
        )
        plt.show()

    def retention_event_type(self, df: pd.DataFrame) -> None:
        grouped = (
            df.groupby(["event_type_main", "is_two"])
            .count()[["event_count"]]
            .unstack()
        )
        self._retention(grouped)

    def retention_device(self, df: pd.DataFrame) -> None:
        grouped = (
            df.groupby(["device_type_canonical", "is_two"])
            .count()[["user_id"]]
            .unstack()
        )
        self._retention(grouped)

    def retention_region(self, df: pd.DataFrame) -> None:
        grouped = (
            df.groupby(["region_name", "is_two"])
            .count()
            .sort_values(by="user_id", ascending=False)[["user_id"]][:10]
            .unstack()
        )
        self._retention(grouped)

    @staticmethod
    def first_hypothesis(df: pd.DataFrame) -> None:
        grouped = df.groupby("event_type_main")
        print(
            pretty_describe(grouped).loc[["спорт", "концерты"], "is_two"][
                ["mean", "count"]
            ]
        )

    @staticmethod
    def second_hypothesis(df: pd.DataFrame) -> None:
        grouped = df.groupby("region_name")
        description = pretty_describe(grouped).sort_values(
            ("event_count", "count"), ascending=False
        )
        description[description["is_two"]["count"] >= 300]["is_two"][
            "mean"
        ].plot(kind="bar", figsize=(10, 6), rot=45)
        plt.xlabel("Регион")
        plt.ylabel("Доля вернувшихся клиентов")
        plt.title("Доля вернувшихся клиентов в зависимости от региона")
        plt.show()
        print(
            description[description["is_two"]["count"] >= 300]["is_two"][
                "mean"
            ]
        )

    @staticmethod
    def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
        df["weekday"] = df.apply(
            lambda row: row["min_order_dt"].day_name(), axis=1
        )
        df["segment"] = pd.cut(
            df["event_count"], bins=[0, 1, 2, 4, float("inf")]
        )
        cormat = df[
            [
                "event_count",
                "min_order_dt",
                "max_order_dt",
                "days_since_prev",
                "mean_revenue_rub",
                "region_name",
                "weekday",
                "event_type_main",
                "device_type_canonical",
                "segment",
                "is_two",
            ]
        ].phik_matrix(
            interval_cols=[
                "event_count",
                "days_since_prev",
                "mean_revenue_rub",
            ]
        )
        result = cormat[["event_count"]].sort_values(
            "event_count", ascending=False
        )
        print(result)
        return result

    @staticmethod
    def cor_heatmap(cormat: pd.DataFrame) -> None:
        sns.heatmap(
            cormat[["event_count"]].sort_values(
                "event_count", ascending=False
            ),
            cmap="Blues",
        )
        plt.show()
