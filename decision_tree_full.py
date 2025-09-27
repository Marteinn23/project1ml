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

from utils import print_header

def decision_tree_with_gridsearch(X: pd.DataFrame, y: pd.DataFrame) -> GridSearchCV:
    """
    A basic decision tree with gridsearch
    """
    X = X.drop("TLD", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # We have to preprocess the categorical data
    #   -> TLD
    # There is not natural ordering, so OneHotEncoding is the first idea
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_cols), # numbers can stay as-is
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",  drop="first"), categorical_cols)
        ],
        remainder="drop"
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])
    
    param_grid = {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [3, 5, 7, 10, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4]
    }
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print_header("DECISION TREE GRID SEARCH RESULTS")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-validation Score: {grid_search.best_score_:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Predicted benign URL (0)", "Predicted phishing URL (1)"],
                yticklabels=["Actual benign (0)", "Actual phishing, (1)"])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

    # Visualizing the tree
    print_header("DECISION TREE VISUALIZER")
    fitted_tree = best_model.named_steps["classifier"]
    preprocessor = best_model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    
    # Get class names (if you have them, otherwise use string representation)
    class_names = [str(cls) for cls in fitted_tree.classes_]
    
    # Create a larger figure for the tree
    plt.figure(figsize=(20, 10))
    
    # Plot the decision tree
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
    
    # Optional: Print some additional tree information
    print(f"Tree Depth: {fitted_tree.get_depth()}")
    print(f"Number of Leaves: {fitted_tree.get_n_leaves()}")
    print(f"Number of Features Used: {np.sum(fitted_tree.feature_importances_ > 0)}")
    
    # Optional: Show feature importance (this requires the preprocessed feature names)
    if len(feature_names) <= 30:  # Only show if not too many features
        importances = fitted_tree.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop Feature Importances:")
        for i in range(min(10, len(indices))):
            if importances[indices[i]] > 0:
                print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    
    return grid_search