import matplotlib.pyplot as plt
import pandas as pd

df_ass = pd.read_csv("../../datasets/bv-web-analytics-associated.csv")
df_prof = pd.read_csv("../../datasets/bv-profiles.csv")

df = df_ass.merge(df_prof, how="inner", on="email")

df["target"] = (~df["purchases"].isna()).astype(int)

referrer_sums = df.groupby("referrer")["target"].sum()
device_sums = df.groupby("device")["target"].sum()
route_sums = df.groupby("route")["target"].sum()

# Referrer plot
plt.bar(referrer_sums.index, referrer_sums.values)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Total Purchases")
plt.title("Purchases per Referrer")
plt.tight_layout()
plt.show()

plt.bar(device_sums.index, device_sums.values)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Total Purchases")
plt.title("Purchases per Device")
plt.tight_layout()
plt.show()

plt.bar(route_sums.index, route_sums.values)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Total Purchases")
plt.title("Purchases per Route")
plt.tight_layout()
plt.show()