import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

df_web = pd.read_csv("datasets/bv-web-analytics.csv")
df_users = pd.read_csv("datasets/bv-profiles.csv")

df_web = df_web.loc[~df_web["email"].isna()]


# During training
preprocessor = ColumnTransformer([
    ('scaler', MinMaxScaler(), ['scroll_pct', 'time_spent']),
    ('onehot', OneHotEncoder(handle_unknown='ignore'), ['referrer', 'device', 'route'])
])

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
])

df = pipeline.fit_transform(df_web)
