import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin

from utils import print_header, SAFE, PHISHING

class TLDFactorizer(BaseEstimator, TransformerMixin):
    """Custom transformer for TLD column factorization"""
    def __init__(self):
        self.tld_categories_ = None
        self.tld_mapping_ = None
    
    def fit(self, X, y):
        if "TLD" in X.columns:
            X_copy = X.copy()
            X_copy["TLD"], self.tld_categories_ = pd.factorize(X_copy["TLD"])
            self.tld_mapping_ = {category: idx for idx, category in enumerate(self.tld_categories_)}
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        if "TLD" in X_copy.columns and self.tld_mapping_ is not None:
            X_copy["TLD"] = X_copy["TLD"].map(self.tld_mapping_).fillna(-1).astype(int)
        return X_copy

class MedianImputer(BaseEstimator, TransformerMixin):
    """Custom transformer for median to deal with missing values"""
    def __init__(self):
        self.median_values_ = None
    
    def fit(self, X, y):
        self.median_values_ = X.median()
        return self
    
    def transform(self, X):
        return X.fillna(self.median_values_)

def create_gaussian_nb_pipeline():
    """Just a simple pipeline wrapper"""
    return Pipeline([
        ("tld_factorizer", TLDFactorizer()),
        ("imputer", MedianImputer()),
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ])

def get_gaussian_nb_param_grid():
    """Parameter grid for GaussianNB grid search"""
    return {
        "classifier__var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1]
    }

def perform_grid_search(pipeline, param_grid, X, y, cv_folds=5):
    """Performs grid search with cross-validation"""
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv_folds, 
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True
    )
    
    grid_search.fit(X, y)
    return grid_search

def plot_hyperparameter_results(grid_search, classifier_name, param_to_plot=None):
    """Plot hyperparameter results"""
    results = pd.DataFrame(grid_search.cv_results_)
    
    if param_to_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(results[f"param_{param_to_plot}"], results["mean_test_score"], "o-", label="Test Score")
        plt.plot(results[f"param_{param_to_plot}"], results["mean_train_score"], "s-", label="Train Score")
        plt.xlabel(param_to_plot)
        plt.ylabel("Accuracy")
        plt.xscale("log") # yea
        plt.title(f'{classifier_name} - Hyperparameter Tuning')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        params = [col for col in results.columns if col.startswith('param_')]
        n_params = len(params)
        
        if n_params == 1:
            plot_hyperparameter_results(grid_search, classifier_name, params[0].replace('param_', ''))
        elif n_params == 2:
            param1, param2 = params
            pivot_table = results.pivot_table(
                values="mean_test_score", 
                index=f"param_{param1}", 
                columns=f"param_{param2}"
            )
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis")
            plt.title(f"{classifier_name} - Hyperparameter Grid Search Results")
            plt.show()

def perform_cross_validation(pipeline, X, y, cv_folds=5):
    """Perform cross-validation and return scores and predictions"""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_scores = []
    all_y_true = []
    all_y_pred = []
    fold_details = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Clone the pipeline for each fold to avoid contamination
        from sklearn.base import clone
        fold_pipeline = clone(pipeline)
        
        # Fit and predict
        fold_pipeline.fit(X_train, y_train)
        y_pred = fold_pipeline.predict(X_test)
        
        # Calculate fold accuracy
        fold_accuracy = accuracy_score(y_test, y_pred)
        cv_scores.append(fold_accuracy)
        
        # Store predictions for overall analysis
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        # Store fold details
        fold_details.append({
            'fold': fold + 1,
            'train_indices': train_idx,
            'test_indices': test_idx,
            'accuracy': fold_accuracy,
            'pipeline': fold_pipeline
        })
    
    return np.array(cv_scores), np.array(all_y_true), np.array(all_y_pred), fold_details

def test_naive_bayes_with_cross_validation(X, y, cv_folds=5):
    """Test Naive Bayes with 5-fold cross-validation instead of train-test split"""
    
    classifiers_config = {
        "GaussianNB": {
            "pipeline": create_gaussian_nb_pipeline(),
            "param_grid": get_gaussian_nb_param_grid()
        },
    }
    
    results = {}
    
    for name, config in classifiers_config.items():
        print_header(f"Testing {name} with {cv_folds}-Fold Cross Validation")
        
        # First perform grid search to find best parameters
        grid_search = perform_grid_search(
            config["pipeline"], 
            config["param_grid"], 
            X, y, 
            cv_folds
        )
        
        best_params = grid_search.best_params_
        best_pipeline = grid_search.best_estimator_
        
        print(f"Best parameters: {best_params}")
        print(f"Best CV accuracy from grid search: {grid_search.best_score_:.4f}")
        
        # Now perform detailed cross-validation with best parameters
        cv_scores, all_y_true, all_y_pred, fold_details = perform_cross_validation(
            best_pipeline, X, y, cv_folds
        )
        
        overall_accuracy = accuracy_score(all_y_true, all_y_pred)
        
        results[name] = {
            "cv_scores": cv_scores,
            "mean_cv_score": np.mean(cv_scores),
            "std_cv_score": np.std(cv_scores),
            "overall_accuracy": overall_accuracy,
            "predictions": all_y_pred,
            "true_labels": all_y_true,
            "classifier": best_pipeline,
            "grid_search": grid_search,
            "best_params": best_params,
            "fold_details": fold_details
        }
        
        print(f"\nCross-Validation Results ({cv_folds}-fold):")
        for fold, score in enumerate(cv_scores, 1):
            print(f"  Fold {fold}: {score:.4f}")
        
        print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        
        print("\nClassification Report (All Folds):")
        print(classification_report(all_y_true, all_y_pred))
        
        # Plot confusion matrix for all folds combined
        plt.figure(figsize=(6, 4))
        cm = confusion_matrix(all_y_true, all_y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name} ({cv_folds}-Fold CV)")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.show()
        
        plot_hyperparameter_results(grid_search, name)
        analyze_predictions_cv(X, all_y_true, all_y_pred, fold_details)

    print_header("NAIVE BAYES CLASSIFIERS COMPARISON")    
    for name, result in results.items():
        print(f"{name:<15}: Mean CV Accuracy: {result['mean_cv_score']:.4f} (+/- {result['std_cv_score']:.4f}), "
              f"Overall: {result['overall_accuracy']:.4f}, "
              f"Params: {result['best_params']}")
    
    best_name = max(results.items(), key=lambda x: x[1]["mean_cv_score"])[0]
    print(f"\nBest classifier: {best_name}")
    
    return results

def analyze_predictions_cv(X, all_y_true, all_y_pred, fold_details):
    """Analyze predictions from cross-validation"""
    
    results_df = X.copy()
    results_df["Actual"] = all_y_true
    results_df["Predicted"] = all_y_pred
    results_df["Correct"] = (all_y_true == all_y_pred)
    
    cm = confusion_matrix(all_y_true, all_y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print_header("CROSS-VALIDATION CONFUSION MATRIX ANALYSIS")
    print(f"True Negatives (Safe correctly classified): {tn}")
    print(f"False Positives (Safe incorrectly classified as Phishing): {fp}")
    print(f"False Negatives (Phishing incorrectly classified as Safe): {fn}")
    print(f"True Positives (Phishing correctly classified): {tp}")
    print()
    
    # Get indices for each classification type
    tp_indices = results_df[(results_df["Actual"] == PHISHING) & (results_df["Predicted"] == PHISHING)].index
    fp_indices = results_df[(results_df["Actual"] == SAFE) & (results_df["Predicted"] == PHISHING)].index
    tn_indices = results_df[(results_df["Actual"] == SAFE) & (results_df["Predicted"] == SAFE)].index
    fn_indices = results_df[(results_df["Actual"] == PHISHING) & (results_df["Predicted"] == SAFE)].index
    
    # Display examples from the first fold for each type
    first_fold_test_indices = fold_details[0]['test_indices']
    display_examples_cv(results_df, tp_indices, first_fold_test_indices, "TRUE POSITIVES", "Phishing links correctly identified as phishing")
    display_examples_cv(results_df, fp_indices, first_fold_test_indices, "FALSE POSITIVES", "Safe links incorrectly classified as phishing")
    display_examples_cv(results_df, tn_indices, first_fold_test_indices, "TRUE NEGATIVES", "Safe links correctly identified as safe")
    display_examples_cv(results_df, fn_indices, first_fold_test_indices, "FALSE NEGATIVES", "Phishing links incorrectly classified as safe")

def display_examples_cv(results_df, indices, fold_indices, title, description):
    """Display examples from cross-validation"""
    print_header(f"{title} - {description}")
    
    # Only show examples from the first fold to avoid duplicates
    fold_indices_set = set(fold_indices)
    fold_specific_indices = [idx for idx in indices if idx in fold_indices_set]
    
    if len(fold_specific_indices) == 0:
        print("No examples found for this category in the first fold.")
        print()
        return
    
    n_examples = min(3, len(fold_specific_indices))
    example_indices = fold_specific_indices[:n_examples]
    df = pd.read_csv("./dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
    
    for i, idx in enumerate(example_indices, 1):
        example = results_df.loc[idx]
        print(f"Example {i} (Fold 1):")
        print(f"  Index: {idx}")
        print(f"  Actual: {example['Actual']} ({'Phishing' if example['Actual'] == PHISHING else 'Safe'})")
        print(f"  Predicted: {example['Predicted']} ({'Phishing' if example['Predicted'] == PHISHING else 'Safe'})")
        
        # Find matching rows in original dataset
        feature_columns = [col for col in example.index if col not in ["Actual", "Predicted", "Correct"]]
        mask = pd.Series(True, index=df.index)
        for col in feature_columns:
            if col in df.columns:
                mask = mask & (df[col] == example[col])
        
        matching_rows = df[mask]
        
        if len(matching_rows) > 0:
            print(f"  URL: {matching_rows.iloc[0]['URL']}")
        else:
            print("  No exact match found in original dataset")
        print()

def test_naive_bayes(X, y):
    """Run complete analysis with 5-fold cross-validation"""
    print_header("5-FOLD CROSS VALIDATION RESULTS")
    print("Using 5-fold cross validation instead of train-test split")
    print(f"Dataset size: {X.shape[0]} samples")
    print(f"Number of features: {X.shape[1]}")
    print()
    
    results = test_naive_bayes_with_cross_validation(X, y, cv_folds=5)    
    
    # Print detailed cross-validation results
    print_header("DETAILED CROSS-VALIDATION SUMMARY")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Mean CV Accuracy: {result['mean_cv_score']:.4f} (+/- {result['std_cv_score']:.4f})")
        print(f"  Fold Accuracies: {[f'{score:.4f}' for score in result['cv_scores']]}")
        print(f"  Overall Accuracy: {result['overall_accuracy']:.4f}")
        print(f"  Best Parameters: {result['best_params']}")
    
    return results