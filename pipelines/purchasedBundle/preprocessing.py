import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from datetime import datetime

def calculate_age(birthdate_series):
    today = datetime.today()
    ages = pd.to_datetime(birthdate_series, format="%d/%m/%Y").apply(
        lambda b: today.year - b.year - ((today.month, today.day) < (b.month, b.day))
    )
    return ages

df_users = pd.read_csv("../../datasets/bv-profiles")
df_web = pd.read_csv("../../datasets/bv-web-analytics-associated")

'''
Features
most-common-route
education
avg-scroll-pct
age

Target
purchases
'''
#Extract logged in users
df_web = df_web.loc[~df_web["email"].isna()].copy()
df_web.groupby("email").agg(
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
).reset_index()

ENCODABLE = ["most-common-route", "education"]

df = df_web.merge(df_users, "inner", "email")



