"""
Authors: Marteinn, Teitur, Tryggvi
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

## classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def main():
    pure_data_set = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

    features = [
        "TLD",
        "TLDLength",
        "URLLength",
        "IsDomainIP",
        "NoOfSubDomain",
        "IsHTTPS",
        "NoOfDegitsInURL",
        "NoOfEqualsInURL",
        "NoOfQMarkInURL",
        "NoOfAmpersandInURL",
        "NoOfOtherSpecialCharsInURL",
    ]


def plot_correlation(data_set):
    drop_cols = ["FILENAME", "URL", "Domain", "Title"]
    data_set = data_set.drop(columns=drop_cols)

    categorical_cols = ["TLD", "Robots"]
    le = LabelEncoder()
    for col in categorical_cols:
        if col in data_set.columns:
            data_set[col] = le.fit_transform(data_set[col].astype(str))

    correlations = data_set.corr()["label"].sort_values(ascending=False)

    # Plot correlations
    plt.figure(figsize=(8, 12))
    sns.barplot(y=correlations.index, x=correlations.values, palette="coolwarm")
    plt.axvline(x=0, color="k", linestyle="--")
    plt.title("Feature Correlations with Label")
    plt.tight_layout()
    plt.show()

    # Heatmap of top correlations
    top_features = correlations.abs().sort_values(ascending=False).head(15).index
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data_set[top_features].corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f"
    )
    plt.title("Top Feature Correlations")
    plt.show()
    return


if __name__ == "__main__":
    main()
