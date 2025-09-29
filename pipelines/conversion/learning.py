import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from yellowbrick.classifier import ClassificationReport

df = pd.read_csv("datasets/aggregated-analytics.csv")

quatil1 = df.loc[df["target"] == 0]["total_time_spent"].quantile(0.25)
quatil3 = df.loc[df["target"] == 0]["total_time_spent"].quantile(0.75)
iinterquartil = quatil3 - quatil1
upper_bound = quatil3 + 1.5 * iinterquartil

# df = df.loc[df["target"] == 1 ^ df["total_time_spent"].lt(upper_bound)]

#sns.barplot(df, x="target", y="experience")
# sns.lineplot
#plt.show()

encodable = ["most_common_referrer", "most_common_device", "experience", "education", "most_common_route"]
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[encodable])
df = pd.concat(
    [
        df,
        pd.DataFrame(encoded, columns=encoder.get_feature_names_out(encodable)),
    ],
    axis=1,
)

df.drop(columns=encodable, inplace=True)
df.drop(columns=["mbti", "interests", "age", "state", "avg_scroll_pct"], inplace=True)



"""
[X]
email
avg_scroll_pct
state
interests

[V]
total_time_spent
most_common_referrer
most_common_device
experience
education
target
most_common_route

[?]
mbti
age
"""

y = df["target"]
x = df.drop("target", axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=25
)

KNN_model = KNeighborsClassifier(n_neighbors=3)
KNN_model.fit(x_train, y_train)

print(KNN_model.score(x_test, y_test))

# TODO: testar com árvore
# TODO: gráfico com diferentes K neighbours

# y_pred = KNN_model.predict(y_test)

# model = KNeighborsClassifier(n_neighbors=3)
# visualizer = ClassificationReport(model, classes=["Não Logado", "Logado"], support=True)

# visualizer.fit(x_train, y_train)        
# visualizer.score(x_test, y_test)        
# visualizer.show()

# labels=["Não Logado", "Logado"]

# ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=labels)

# plt.show()