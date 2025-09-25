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
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from https_model import simple_https_guesser
from naive_bayes import test_naive_bayes

DROP_COLS = ["FILENAME", "URL", "Domain", "Title"]

def main():
    pure_data_set = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
    pure_data_set = pure_data_set.drop(columns=DROP_COLS)
    pure_data_set["label"] = pure_data_set["label"].map({0: 1, 1: 0})

    # CORRECT WAY: Count rows where label=0 AND IsHTTPS=1
    count_https_phishing = len(pure_data_set[(pure_data_set["label"] == 0) & (pure_data_set["IsHTTPS"] == 1)])
    print(f"Number of phishing sites (label=0) that use HTTPS: {count_https_phishing}")
    count_http_phishing = len(pure_data_set[(pure_data_set["label"] == 1) & (pure_data_set["IsHTTPS"] == 1)])
    print(f"Number of phishing sites (label=0) that use HTTPS: {count_http_phishing}")


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
    X = pure_data_set[features]
    y = pure_data_set["label"]

    # split into train/test
    # careful to only use the test sets when we're ready!
    X_train ,__X_test, y_train, __y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        train_size=0.8,
        random_state=42,
    )

    # Predict phishing if http. Else predict legitimate
    simple_https_guesser(X_train, y_train)
    results = test_naive_bayes(X_train, y_train)

    # Try again, but with all of the features.
    features = [col for col in pure_data_set.columns if col != "label"]
    X = pure_data_set[features]
    y = pure_data_set["label"]

    X_train ,__X_test, y_train, __y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        train_size=0.8,
        random_state=42,
    )

    simple_https_guesser(X_train, y_train)
    results = test_naive_bayes(X_train, y_train)


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
