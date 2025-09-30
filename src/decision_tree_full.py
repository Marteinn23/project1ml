import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from utils import print_header, plot_confusion_matrix


def decision_tree_with_gridsearch(
    X: pd.DataFrame, y: pd.DataFrame, drop=True
) -> GridSearchCV:
    """
    A basic decision tree with gridsearch
    """
    if drop:
        X = X.drop("TLD", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Honestly this is mostly done for the tree visualization
    # This does no sort of preprocessing on the data, everything is numerical at this stage
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_cols),
            ("cat", OneHotEncoder(), categorical_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ]
    )

    param_grid = {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [3, 5, 7, 10, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print_header("DECISION TREE GRID SEARCH RESULTS")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-validation Score: {grid_search.best_score_:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    plot_confusion_matrix(y_test, y_pred, "Decision tree")

    print_header("DECISION TREE VISUALIZER")
    fitted_tree = best_model.named_steps["classifier"]
    preprocessor = best_model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    class_names = [str(cls) for cls in fitted_tree.classes_]
    plt.figure(figsize=(20, 10))
    plot_tree(
        fitted_tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        proportion=True,
        fontsize=10,
        max_depth=3,
    )

    plt.title(f"Decision Tree (Max Depth: {fitted_tree.get_depth()})")
    plt.tight_layout()
    plt.show()

    print(f"Tree Depth: {fitted_tree.get_depth()}")
    print(f"Number of Leaves: {fitted_tree.get_n_leaves()}")

    return grid_search
