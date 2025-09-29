import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df_signUP = pd.read_csv("datasets/preprocessed-signup.csv")

df_new = df_signUP.drop(
    [
        "route_/",
        "route_/cursos",
        "route_/instituicoes",
        "route_/quiz",
        "route_/vestibular/ENEM",
        "route_/vestibular/FATEC",
        "route_/vestibular/FUVEST",
        "route_/vestibular/UNICAMP",
    ],
    axis=1,
)

y = df_new["target"]
x = df_new.drop("target", axis=1)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=25
)

treeModel = DecisionTreeClassifier(random_state=2142, max_depth=5)

treeModel.fit(x_train, y_train)

print(treeModel.score(x_test, y_test))

y_pred = treeModel.predict(x_test)
print(classification_report(y_test, y_pred))
