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


DROP_COLS = ["FILENAME", "URL", "Domain", "Title"]


def main():
    pure_data_set = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
    pure_data_set = pure_data_set.drop(columns=DROP_COLS)

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
        "DomainLength",
    ]
    # X = input features, y = target column
    X = pure_data_set[features]
    y = pure_data_set["label"]  # <-- this is your target

    # split into train/test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        train_size=0.8,
        random_state=42,  # optional but useful for reproducibility
    )
    plot_correlation(X_test, Y_test)


def plot_correlation(data_set, labels):
    categorical_cols = ["TLD", "Robots"]
    le = LabelEncoder()
    for col in categorical_cols:
        if col in data_set.columns:
            data_set[col] = le.fit_transform(data_set[col].astype(str))

    # combine features + label so correlation works
    df = data_set.copy()
    df["label"] = labels

    correlations = df.corr()["label"].sort_values(ascending=False)

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
