import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from utils import print_header, PHISHING, SAFE

def simple_https_guesser(X, y):
    """
    A very simple model that uses just one explanatory variable:
        is it https or not?

    if IsHTTPS -> 1, legitimate
    if IsHTTPS -> 0, phishing
    """
    X_https = X["IsHTTPS"]
    
    # Make predictions: HTTPS=1 → legitimate (0), HTTPS=0 → phishing (1)
    y_pred = (X_https == 1).astype(int).map({0: 1, 1: 0})
    accuracy = accuracy_score(y, y_pred)

    print_header("SIMPLE HTTPS GUESSER RESULTS")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"HTTPS=0 (not secure) → Phishing (0)")
    print(f"HTTPS=1 (secure) → Legitimate (1)")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Fix confusion matrix labels
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
    )
    plt.title("Confusion Matrix - Simple HTTPS Guesser")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()

    return y_pred