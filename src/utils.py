import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score

PHISHING = 1
SAFE = 0

def print_header(header):
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")

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