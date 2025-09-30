import re
from datetime import datetime

import pandas as pd
from pandas import DataFrame as DF
from sklearn.preprocessing import OneHotEncoder


def extract_base_route(route) -> str | None:
    if pd.isna(route):
        return None
    match = re.match(r"^(/[^/]+)", str(route))
    return match.group(1) if match else route


def calculate_age(birthdate_series):
    today = datetime.today()
    ages = pd.to_datetime(birthdate_series, format="%d/%m/%Y").apply(
        lambda b: today.year - b.year - ((today.month, today.day) < (b.month, b.day))
    )
    return ages


def aggregate_analytics_data(df_web) -> DF:
    df_web = df_web.loc[~df_web["email"].isna()].copy()
    df_web["base_route"] = df_web["route"].apply(extract_base_route)

    return (
        df_web.groupby("email")
        .agg(
            avg_scroll_pct=("scroll_pct", "mean"),
            total_time_spent=("time_spent", "sum"),
            most_common_referrer=(
                "referrer",
                lambda x: x.mode()[0] if not x.mode().empty else None,
            ),
            most_common_device=(
                "device",
                lambda x: x.mode()[0] if not x.mode().empty else None,
            ),
            most_common_route=(
                "base_route",
                lambda x: x.mode()[0] if not x.mode().empty else None,
            ),
        )
        .reset_index()
    )


ENCODABLE = [
    "most_common_referrer",
    "most_common_device",
    "experience",
    "education",
    "most_common_route",
]
UNUSED = [
    "mbti",
    "interests",
    "age",
    "state",
    "avg_scroll_pct",
    "birthdate",
    "purchases",
    "email",
]

if __name__ == "__main__":
    df_web = pd.read_csv("datasets/bv-web-analytics-associated.csv")
    df_users = pd.read_csv("datasets/bv-profiles.csv")

    df_agg = aggregate_analytics_data(df_web)

    df = df_agg.merge(df_users, how="inner", on="email")
    df["age"] = calculate_age(df["birthdate"])
    df["target"] = ~df["purchases"].isna()

    non_purchase_time = df.loc[df["target"] == 0]["total_time_spent"]
    q1 = non_purchase_time.quantile(0.25)
    q3 = non_purchase_time.quantile(0.75)
    interquartile = q3 - q1
    upper_bound = q3 + 1.5 * interquartile
    df = df.loc[(df["target"] == 1) | (df["total_time_spent"] < upper_bound)]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[ENCODABLE])
    encoded_df = DF(encoded, df.index, encoder.get_feature_names_out(ENCODABLE))
    df = pd.concat([df, encoded_df], axis=1)

    # BACKLOG: decouple mbti

    df.drop(columns=ENCODABLE + UNUSED, inplace=True)

    df.to_csv("datasets/aggregated-analytics.csv")
