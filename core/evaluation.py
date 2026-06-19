from sklearn.metrics import silhouette_score, davies_bouldin_score

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score
)


def evaluate_classification(model, X_test, y_test):

    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average='weighted', zero_division=0),
        "recall": recall_score(y_test, predictions, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test, predictions, average='weighted', zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions)
    }


def format_classification_results(results):

    return f"""
===== Classification Evaluation =====

Accuracy  : {results['accuracy']:.4f}
Precision : {results['precision']:.4f}
Recall    : {results['recall']:.4f}
F1-Score  : {results['f1_score']:.4f}

Confusion Matrix:
{results['confusion_matrix']}
"""


def evaluate_clustering(model, data, clusters):

    return {
        "silhouette_score": silhouette_score(data, clusters),
        "inertia": model.inertia_
    }


def format_clustering_results(results):

    return f"""
===== Clustering Evaluation =====

Silhouette Score : {results['silhouette_score']:.4f}
Inertia          : {results['inertia']:.4f}
"""

def evaluate_clustering(model, data, clusters):

    return {
        "silhouette_score": silhouette_score(data, clusters),
        "inertia": model.inertia_,
        "davies_bouldin": davies_bouldin_score(data, clusters)
    }


def format_clustering_results(results):

    return f"""
===== Clustering Evaluation =====

Silhouette Score : {results['silhouette_score']:.4f}
Inertia          : {results['inertia']:.4f}
Davies-Bouldin   : {results['davies_bouldin']:.4f}
"""