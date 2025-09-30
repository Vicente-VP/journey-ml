import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from yellowbrick.classifier import ClassificationReport

df = pd.read_csv("datasets/aggregated-analytics.csv")

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

y_pred = KNN_model.predict(y_test)

model = KNeighborsClassifier(n_neighbors=3)
visualizer = ClassificationReport(model, classes=["Não Logado", "Logado"], support=True)

visualizer.fit(x_train, y_train)
visualizer.score(x_test, y_test)
visualizer.show()

labels=["Não Logado", "Logado"]

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=labels)

plt.show()
