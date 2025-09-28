import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import VotingClassifier

from baseline_model import baseline_model
from decision_tree_full import decision_tree_with_gridsearch
from models import PHISURL_NaiveBayes, PHISURL_NeuralNetwork, PHISURL_RandomForest

from utils import plot_confusion_matrix


DROP_COLS = ["FILENAME", "URL", "Domain", "Title"]

def main():
    pure_data_set = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
    pure_data_set = pure_data_set.drop(columns=DROP_COLS)
    # 0 to 1 and 1 to 0
    # This means that Phishing is 1
    # Safe is 0
    pure_data_set["label"] = pure_data_set["label"].map({0: 1, 1: 0})

    X = pure_data_set.drop("label", axis=1)
    y = pure_data_set["label"]

    X_train , X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        train_size=0.8,
        random_state=42,
    )

    # baseline model
    if input("baseline? ") == "y":
        baseline_model(X_train, y_train)

    # A basic decision tree grid-search
    if input("Decision tree? ") == "y":
        decision_tree_with_gridsearch(X_train, y_train)

    # That was a lot better than expected.
    # Clearly the URL similarity index is the main workhorse
    # We want our model to work on a straight up URL
    #   -> The URL similarity index needs data on legitamate URLs and more
    # Let's constrain the model and see how it does on only basic URL variables

    # Top level domains are very diverse, and can cause an explosion in dimensions if we use one-hot encoding
    # We propose to rather count their frequency, and if they are a common TLD (https://en.wikipedia.org/wiki/List_of_Internet_top-level_domains)
    def extract_tld_features(df):
        tld_freq = df["TLD"].value_counts(normalize=True)
        df["TLDFrequency"] = df["TLD"].map(tld_freq).fillna(0) # any NaN to 0
        
        return df

    X = extract_tld_features(X)

    features = [
        "TLDFrequency",
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


    X_constrained = X[features]
    X_train_constrained, X_test_constrained, y_train_constrained, y_test_constrained = train_test_split(
        X_constrained,
        y,
        test_size=0.2,
        train_size=0.8,
        random_state=42,
    )

    # Individual model grid searches
    best_models = {}
    
    final_classifier = None
    if input("Decision tree constrained? ") == "y":
        decision_tree_with_gridsearch(X_train_constrained, y_train_constrained, False)
    if input("Random Forest grid search? ") == "y":
        best_models["rf"] = random_forest_gridsearch(X_train_constrained, y_train_constrained, X_test_constrained, y_test_constrained)
    
    if input("Neural Network grid search? ") == "y":
        best_models["nn"] = neural_network_gridsearch(X_train_constrained, y_train_constrained, X_test_constrained, y_test_constrained)
    
    if input("Naive Bayes grid search? ") == "y":
        best_models["nb"] = naive_bayes_gridsearch(X_train_constrained, y_train_constrained, X_test_constrained, y_test_constrained)

    # Voting classifier with best models
    if input("Voting classifier grid search? ") == "y":
        final_classifier = voting_classifier_gridsearch(X_train_constrained, y_train_constrained, X_test_constrained, y_test_constrained, best_models)

    # Now with out final tuned classifier we can check how it handles unseen data.
    pure_data_set = pd.read_csv("dataset/proccessed_urls.csv").drop("URL", axis=1)
    pure_data_set["Label"] = pure_data_set["Label"].map({0: 1, 1: 0})
    X = extract_tld_features(pure_data_set)
    X = X[features]
    y = pure_data_set["Label"]

    y_pred = final_classifier.predict(X)
    plot_confusion_matrix(y, y_pred)


def random_forest_gridsearch(X_train, y_train, X_test, y_test):
    """Perform grid search for Random Forest model"""
    print("Performing Random Forest grid search...")
    
    rf_param_grid = {
        "n_estimators": [25, 50, 100],
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 8]
    }
    
    rf_model = PHISURL_RandomForest()
    
    grid = GridSearchCV(
        rf_model,
        param_grid=rf_param_grid,
        scoring=make_scorer(fn_focused_scorer),
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    print("Random Forest - Best parameters:", grid.best_params_)
    print("Random Forest - Best score:", grid.best_score_)

    y_pred = grid.predict(X_test)
    cm = plot_confusion_matrix(y_test, y_pred, "random forest")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return grid.best_estimator_


def neural_network_gridsearch(X_train, y_train, X_test, y_test):
    """Grid search for Neural Network model"""
    print("Performing Neural Network grid search...")
    
    nn_param_grid = {
        "hidden_layer_sizes": [
            [10],
            [10, 20],
            [25, 50],
            [50],
            [50, 25],
            [100],
            [100, 50]
        ],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate_init": [0.001, 0.01, 0.1]
    }
    
    nn_model = PHISURL_NeuralNetwork()
    
    grid = GridSearchCV(
        nn_model,
        param_grid=nn_param_grid,
        scoring=make_scorer(fn_focused_scorer),
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    print("Neural Network - Best parameters:", grid.best_params_)
    print("Neural Network - Best score:", grid.best_score_)

    y_pred = grid.predict(X_test)
    cm = plot_confusion_matrix(y_test, y_pred, "neural network")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return grid.best_estimator_


def naive_bayes_gridsearch(X_train, y_train, X_test, y_test):
    """Grid search for Naive Bayes model"""
    print("Performing Naive Bayes grid search...")
    nb_param_grid = {
        "alpha": [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]
    }
    
    nb_model = PHISURL_NaiveBayes()
    
    grid = GridSearchCV(
        nb_model,
        param_grid=nb_param_grid,
        scoring=make_scorer(fn_focused_scorer),
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    print("Naive Bayes - Best parameters:", grid.best_params_)
    print("Naive Bayes - Best score:", grid.best_score_)

    y_pred = grid.predict(X_test)
    cm = plot_confusion_matrix(y_test, y_pred, "naive bayes bernoulli")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return grid.best_estimator_


def voting_classifier_gridsearch(X_train, y_train, X_test, y_test, best_models):
    """Grid search for Voting Classifier"""
    print("Performing Voting Classifier grid search...")
    
    # Create ensemble with best models
    ensemble = VotingClassifier(
        estimators=[
            ("nn", best_models["nn"]),
            ("nb", best_models["nb"]),
            ("rf", best_models["rf"]),
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
            [1, 0, 0], # Use only NN
            [0, 1, 0], # Use only NB
            [0, 0, 1], # Use only RF
        ]
    }

    grid = GridSearchCV(
        ensemble,
        param_grid=param_grid,
        scoring=fn_scorer,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    print("Voting Classifier - Best parameters:", grid.best_params_)
    print("Voting Classifier - Best score:", grid.best_score_)

    y_pred = grid.predict(X_test)
    cm = plot_confusion_matrix(y_test, y_pred, "Voting Ensemble")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return grid.best_estimator_


def fn_focused_scorer(y_true, y_pred, fn_weight=5.0, fp_weight=1.0):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    weighted_score = (fn_weight * recall + fp_weight * specificity) / (fn_weight + fp_weight)    
    return weighted_score


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