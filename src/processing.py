from dataclasses import dataclass
from typing import ClassVar
import pandas as pd
from sqlalchemy import create_engine


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

    def preprocess(self) -> pd.DataFrame:
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
