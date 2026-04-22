from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)


def compute_metrics(y_test, y_pred, y_prob=None):
    # collect all metrics into a dict for easy access
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
    }

    # auc needs probability scores, not all models provide this
    if y_prob is not None:
        results['auc'] = roc_auc_score(y_test, y_prob)

    return results


def print_metrics(name, results):
    # formatted output for each model's results
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")
    if 'auc' in results:
        print(f"  AUC:       {results['auc']:.4f}")
    print(f"\n  Confusion Matrix:\n{results['confusion_matrix']}")
    print(f"\n{results['classification_report']}")
