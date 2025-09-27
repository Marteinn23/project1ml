"""
Authors: Marteinn, Teitur, Tryggvi
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn modules
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, recall_score, confusion_matrix
from sklearn.ensemble import VotingClassifier

from baseline_model import baseline_model
from decision_tree_full import decision_tree_with_gridsearch
from models import PHISURL_NaiveBayes, PHISURL_NeuralNetwork, PHISURL_RandomForest


DROP_COLS = ["FILENAME", "URL", "Domain", "Title"]

def main():
    pure_data_set = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
    pure_data_set = pure_data_set.drop(columns=DROP_COLS)
    pure_data_set["label"] = pure_data_set["label"].map({0: 1, 1: 0})


    X = pure_data_set.drop("label", axis=1)
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

    # baseline model
    # baseline_model(X_train, y_train)

    # A basic decision tree grid-search
    # decision_tree_with_gridsearch(X_train, y_train)

    # That was a lot better than expected.
    # Clearly the URL similarity index is the main workhorse
    # We want our model to work on a straight up URL
    #   -> The URL similarity index needs data on legitamate URLs and more
    # Let's constrain the model and see how it does on only basic URL variables
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

    X_constrained = X[features]
    X_constrained = X_constrained.drop("TLD", axis=1) # TODO
    X_train_constrained, __X_test_constrained, y_train_constrained, __y_test_constrained = train_test_split(
        X_constrained,
        y,
        test_size=0.2,
        train_size=0.8,
        random_state=42,
    )

    # Start by basic grid search:
    # on random forest
    # on neural network
    # on bernoulli NB


    # combine into one ensemble:
    # Use the best hyperparameters we found from the first 3 models
    # do grid search for ensemble on proba_threshold to minimize false negatives
    # and on weights for each model
    ensemble = VotingClassifier(
        estimators=[
            ("nn", PHISURL_NeuralNetwork()),
            ("nb", PHISURL_NaiveBayes()),
            ("rf", PHISURL_RandomForest())
        ],
        voting="soft"
    )

    fn_scorer = make_scorer(fn_focused_scorer, greater_is_better=True)

    param_grid = {
        "weights": [
            [1, 1, 1], # Equal weights
            [2, 1, 1], # Favor NN
            [1, 2, 1], # Favor NB  
            [1, 1, 2], # Favor RF
            [3, 1, 1], # Strongly favor NN
            [1, 3, 1], # Strongly favor NB
            [1, 1, 3], # Strongly favor RF
        ]
    }

    grid = GridSearchCV(
        ensemble,
        param_grid=param_grid,
        scoring=fn_scorer, # Maybe just grid-search this with a for-loop...
        cv=5,
        n_jobs=-1
    )
    
    grid.fit(X_train_constrained, y_train_constrained)
    print("Best parameters:", grid.best_params_)
    print("Best score:", grid.best_score_)

    # Get predictions on test set
    y_pred = grid.predict(__X_test_constrained)
    y_pred_proba = grid.predict_proba(__X_test_constrained)

    # Plot confusion matrix
    cm = plot_confusion_matrix(__y_test_constrained, y_pred, "Voting Ensemble")


def fn_focused_scorer(y_true, y_pred, fn_weight=10.0, fp_weight=1.0):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    weighted_score = (fn_weight * recall + fp_weight * specificity) / (fn_weight + fp_weight)    
    return weighted_score


def plot_confusion_matrix(y_true, y_pred, model_name=""):
    """
    Plot a BEAUTIFUL confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Phishing", "Benign"],
                yticklabels=["Phishing", "Benign"])
    
    plt.title(f'Confusion Matrix - {model_name}\n(Recall: {recall_score(y_true, y_pred):.3f})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
    
    return cm

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
