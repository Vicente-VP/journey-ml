import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ClassificationReport

DATA_PATH = "datasets/course_recommendation_interests.csv"
MODEL_OUTPUT_PATH = "models/course_recommendation_model.pkl"
TARGET_COL = "target_course"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Greens", values_format="d")
    plt.title(f"Confusion Matrix - Random Forest", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=8):
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], color="steelblue")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Feature Importances - Random Forest")
    plt.tight_layout()
    plt.show()
    plt.close()


def train_model():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    classes = sorted(y.unique())

    visualizer = ClassificationReport(model, classes=classes, support=True)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show()

    plot_confusion_matrix(y_test, y_pred, classes)

    plot_feature_importance(model, feature_names)

    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_names": feature_names,
                "classes": classes,
                "model_type": "random_forest",
                "accuracy": accuracy,
            },
            f,
        )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classes: {classes}")

    return model, X_test, y_test, y_pred


if __name__ == "__main__":
    model, X_test, y_test, y_pred = train_model()
