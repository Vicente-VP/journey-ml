import re
from datetime import datetime
import dateutil

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df_web = pd.read_csv("datasets/bv-web-analytics-associated.csv")
df_users = pd.read_csv("datasets/bv-profiles.csv")

df_web = df_web.loc[~df_web["email"].isna()]


def extract_base_route(route):
    if pd.isna(route):
        return None
    match = re.match(r"^(/[^/]+)", str(route))
    return match.group(1) if match else route


df_web["base_route"] = df_web["route"].apply(extract_base_route)

# Group by email and aggregate
df_web_agg = (
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

df = df_web_agg.merge(df_users, how="inner", on="email")

today = datetime.today()
df["age"] = pd.to_datetime(df["birthdate"], format="%d/%m/%Y").apply(
    lambda b: today.year - b.year - ((today.month, today.day) < (b.month, b.day))
)

# BACKLOG: decouple mbti

df["target"] = ~df["purchases"].isna()
df.drop(columns=["birthdate", "purchases", "email"], inplace=True)


# TODO: onehot most_common_referrer,most_common_device,most_common_route
# TODO: onehot education,state,interests,experience
# TODO: purchases -> target

df.to_csv("datasets/aggregated-analytics.csv")
