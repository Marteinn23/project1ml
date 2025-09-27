import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix
from utils import print_header

def baseline_model(X: pd.DataFrame, y: pd.DataFrame):
    """
    # The baseline model for comparisons
    Any model made should outperform this
    """
    
    # Since this is a binary classification, we can take a sum and choose the more frequent label
    average = y.mean().item()
    if (average < 0.5):
        guess = 0
    else:
        guess = 1

    # Just check if our guess is correct or not
    y_pred = pd.Series([guess] * len(y))
    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    print_header("BASELINE MODEL")
    print(f"Accuracy: {accuracy}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, fmt="d", cmap="Blues",
                xticklabels=["Predicted benign URL (0)", "Predicted phishing URL (1)"],
                yticklabels=["Actual benign (0)", "Actual phishing, (1)"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix - Baseline model")
    plt.show()

