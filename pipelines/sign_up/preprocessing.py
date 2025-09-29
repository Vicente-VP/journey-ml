import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


matplotlib.use("TkAgg")

df_web = pd.read_csv("./datasets/bv-web-analytics.csv")

df_signup = df_web.copy()
df_signup.rename(columns={"email": "target"}, inplace=True)
df_signup["target"] = (~df_signup["target"].isna()).astype(int)

df_signup["month"] = pd.to_datetime(df_signup["date"], format='%d/%m/%Y').dt.month

sns.barplot(df_signup[["target", "month"]], x="month", y="target")
plt.show()

encodable = ["referrer", "device", "route", "month"]
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df_signup[encodable])
df_signup = pd.concat(
    [
        df_signup,
        pd.DataFrame(encoded, columns=encoder.get_feature_names_out(encodable)),
    ],
    axis=1,
)

df_signup.drop(columns=encodable, inplace=True)

scalable = ["scroll_pct", "time_spent"]
scaler = MinMaxScaler()
df_signup[scalable] = scaler.fit_transform(df_signup[scalable])

df_signup.drop(columns=["ip", "date"], inplace=True)


print(df_signup["target"].sum())

df_signup.to_csv("./datasets/preprocessed-signup.csv", index=False)
