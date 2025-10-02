"""
training.py - Train course recommendation model with evaluation metrics
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay
)

# Configuration
DATA_PATH = "datasets/course_recommendation_interests.csv"
MODEL_OUTPUT_PATH = "models/course_recommendation_model.pkl"
TARGET_COL = "target_course"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Choose model: 'decision_tree', 'random_forest', or 'knn'
MODEL_TYPE = 'decision_tree'


# ============================================================================
# MODEL SELECTION
# ============================================================================

def get_model(model_type):
    """Return the selected model"""
    models = {
        'decision_tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            random_state=RANDOM_STATE
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=20,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'knn': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        )
    }
    return models[model_type]


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, classes, save_path="confusion_matrix.png"):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - {MODEL_TYPE.replace("_", " ").title()}', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Confusion matrix saved to {save_path}")
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20, save_path="feature_importance.png"):
    """Plot feature importance for tree-based models"""
    if not hasattr(model, 'feature_importances_'):
        print("  Model does not support feature importance")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances - {MODEL_TYPE.replace("_", " ").title()}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Feature importance plot saved to {save_path}")
    plt.show()


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_model():
    """Main training workflow"""
    
    print("=" * 70)
    print("COURSE RECOMMENDATION MODEL TRAINING")
    print("=" * 70)
    
    # 1. Load data
    print(f"\n[1/6] Loading preprocessed data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Target distribution:\n{df[TARGET_COL].value_counts()}")
    
    # 2. Split features and target
    print("\n[2/6] Splitting features and target...")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    feature_names = X.columns.tolist()
    
    print(f"  Features: {X.shape[1]} columns")
    print(f"  Target classes: {y.nunique()} unique courses")
    
    # 3. Train-test split
    print(f"\n[3/6] Splitting into train/test sets ({int((1-TEST_SIZE)*100)}%/{int(TEST_SIZE*100)})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # 4. Train model
    print(f"\n[4/6] Training {MODEL_TYPE.replace('_', ' ').title()} model...")
    model = get_model(MODEL_TYPE)
    model.fit(X_train, y_train)
    print("  ✓ Training complete!")
    
    # 5. Evaluate
    print("\n[5/6] Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification Report
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    report = classification_report(y_test, y_pred)
    print(report)
    
    # Confusion Matrix
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    classes = sorted(y.unique())
    plot_confusion_matrix(y_test, y_pred, classes)
    
    # Feature Importance (if applicable)
    if MODEL_TYPE in ['decision_tree', 'random_forest']:
        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE")
        print("=" * 70)
        plot_feature_importance(model, feature_names)
    
    # 6. Save model
    print("\n[6/6] Saving trained model...")
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': feature_names,
            'classes': classes,
            'model_type': MODEL_TYPE,
            'accuracy': accuracy
        }, f)
    print(f"  ✓ Model saved to {MODEL_OUTPUT_PATH}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Model Type: {MODEL_TYPE.replace('_', ' ').title()}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Number of Classes: {len(classes)}")
    print(f"Classes: {classes}")
    print("=" * 70 + "\n")
    
    return model, X_test, y_test, y_pred


# ============================================================================
# PREDICTION EXAMPLE
# ============================================================================

def load_and_predict_example():
    """Example of how to load the saved model and make predictions"""
    print("\n" + "=" * 70)
    print("PREDICTION EXAMPLE")
    print("=" * 70)
    
    # Load model
    with open(MODEL_OUTPUT_PATH, 'rb') as f:
        saved_data = pickle.load(f)
    
    model = saved_data['model']
    feature_names = saved_data['feature_names']
    classes = saved_data['classes']
    
    print(f"Loaded {saved_data['model_type']} model")
    print(f"Model accuracy: {saved_data['accuracy']:.4f}")
    
    # Load test data for example
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    
    # Make predictions on first 5 samples
    sample = X.head(5)
    predictions = model.predict(sample)
    
    # Get prediction probabilities
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(sample)
        
        print("\nSample predictions with probabilities:")
        for i, (pred, proba) in enumerate(zip(predictions, probas)):
            print(f"\nSample {i+1}:")
            print(f"  Predicted: {pred}")
            print(f"  Top 3 courses:")
            top_3_idx = np.argsort(proba)[-3:][::-1]
            for idx in top_3_idx:
                print(f"    - {classes[idx]}: {proba[idx]:.4f}")
    else:
        print("\nSample predictions:")
        for i, pred in enumerate(predictions):
            print(f"  Sample {i+1}: {pred}")
    
    print("=" * 70 + "\n")


# ============================================================================
# EXECUTE
# ============================================================================

if __name__ == "__main__":
    # Train model
    model, X_test, y_test, y_pred = train_model()
    
    # Show prediction example
    load_and_predict_example()