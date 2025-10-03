from datetime import datetime
import pandas as pd
from pandas import DataFrame as DF
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

PROFILES_PATH = "datasets/bv-profiles.csv"
OUTPUT_PATH = "datasets/course_recommendation_interests.csv"

ENCODABLE_COLS = [
    "experience",
    "education",
    "state",
]

UNUSED_COLS = ["birthdate", "email", "mbti"]


def calculate_age(birthdate_series):
    today = datetime.today()
    ages = pd.to_datetime(birthdate_series, format="%d/%m/%Y").apply(
        lambda b: today.year - b.year - ((today.month, today.day) < (b.month, b.day))
    )
    return ages


def explode_purchases(df):
    df = df[df["purchases"].notna()].copy()
    df["target_course"] = df["purchases"].str.split("|")
    df = df.explode("target_course").drop(columns=["purchases"]).reset_index(drop=True)
    return df


def encode_interests(df):
    df["interests_list"] = df["interests"].str.split("|")

    mlb = MultiLabelBinarizer()
    interests_encoded = mlb.fit_transform(df["interests_list"])

    interests_df = DF(
        interests_encoded,
        df.index,
        [f"interest_{interest}" for interest in mlb.classes_],
    )

    df = pd.concat([df, interests_df], axis=1)
    df.drop(columns=["interests", "interests_list"], inplace=True)

    return df, mlb


def encode_categorical_features(df, encodable_cols):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[encodable_cols])
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(encodable_cols), index=df.index
    )

    df = pd.concat([df, encoded_df], axis=1)
    df.drop(columns=encodable_cols, inplace=True)

    return df, encoder


def preprocess_data():
    df = pd.read_csv(PROFILES_PATH)

    print("[1/4] Engineering features...")
    df["age"] = calculate_age(df["birthdate"])
    print(f"Calculated age (range: {df['age'].min()}-{df['age'].max()})")

    print("[2/4] Encoding interests with MultiLabelBinarizer...")
    df, mlb = encode_interests(df)

    print("[3/4] Exploding purchases into individual target courses...")
    df = explode_purchases(df)
    print(f"Targets: {df['target_course'].unique()}")

    print("[4/4] One-hot encoding remaining categorical features...")
    df, encoder = encode_categorical_features(df, ENCODABLE_COLS)

    df.drop(columns=UNUSED_COLS, inplace=True)
    df.to_csv(OUTPUT_PATH, index=False)

    return df, encoder, mlb


if __name__ == "__main__":
    df_processed, encoder, mlb = preprocess_data()
