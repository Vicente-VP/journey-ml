import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from yellowbrick.classifier import ClassificationReport
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("../../datasets/bv-profiles.csv")

today = datetime.today()
df["age"] = pd.to_datetime(df["birthdate"], format="%d/%m/%Y").apply(
        lambda b: today.year - b.year - ((today.month, today.day) < (b.month, b.day))
    )

df["purchases"] = df["purchases"].fillna("")
df["purchases_list"] = df["purchases"].apply(lambda x: x.split("|") if x else [])

mlb = MultiLabelBinarizer()
y = pd.DataFrame(mlb.fit_transform(df["purchases_list"]), columns=mlb.classes_)

df["interests"] = df["interests"].fillna("")
df["interests_list"] = df["interests"].apply(lambda x: x.split("|") if x else [])

mlb_interests = MultiLabelBinarizer()
interests_encoded = pd.DataFrame(mlb_interests.fit_transform(df["interests_list"]), 
                                 columns=[f"interest_{c}" for c in mlb_interests.classes_])

categorical_fields = ["education", "state", "mbti", "experience"]

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
categorical_encoded = encoder.fit_transform(df[categorical_fields])

categorical_encoded = pd.DataFrame(categorical_encoded, 
                                   columns=encoder.get_feature_names_out(categorical_fields))

x = pd.concat([
    df[["age"]].reset_index(drop=True),
    categorical_encoded.reset_index(drop=True),
    interests_encoded.reset_index(drop=True)
], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=25
)

KNN_model = KNeighborsClassifier(n_neighbors=3)
KNN_model.fit(x_train, y_train)

print(KNN_model.score(x_test, y_test))

# TODO: testar com árvore
# TODO: gráfico com diferentes K neighbours

y_pred = KNN_model.predict(x_test)

model = KNeighborsClassifier(n_neighbors=3)

# labels=["Não Comprou", "Comprou"]
# visualizer = ClassificationReport(KNN_model, classes=["Comprou", "Não Comprou"], support=True)

# visualizer.fit(x_train, y_train)
# visualizer.score(x_test, y_test)
# visualizer.show()

# ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=labels)

# plt.show()
