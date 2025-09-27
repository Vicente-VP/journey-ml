import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df_signUP = pd.read_csv("datasets/encoded-signup.csv")
df_web = pd.read_csv("./datasets/bv-web-analytics.csv")
df_web.rename(columns={"email": "target"}, inplace=True)
df_web["target"] = (~df_web["target"].isna())

device_target_counts = df_web.groupby('device')['target'].value_counts(normalize=True).unstack()

referrer_target_counts = df_web.groupby('referrer')['target'].value_counts(normalize=True).unstack()

device_target_counts.plot(kind='bar', figsize=(10,6))
referrer_target_counts.plot(kind='bar', figsize=(10,6))
plt.show()

# df_new = df_signUP.drop(["route_/", "route_/cursos", "route_/instituicoes", "route_/quiz", "route_/vestibular/ENEM", "route_/vestibular/FATEC", "route_/vestibular/FUVEST", "route_/vestibular/UNICAMP"], axis=1)

# y = df_new["target"]
# x = df_new.drop('target', axis=1)


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=25)

# treeModel = DecisionTreeClassifier(random_state=2142, max_depth=5)

# treeModel.fit(x_train, y_train)

# print(treeModel.score(x_test, y_test))