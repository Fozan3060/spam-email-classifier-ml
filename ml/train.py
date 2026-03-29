import os
from feature_extraction import extract_features, vectorizer
from evaluate import evaluate_model, tune_and_evaluate
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from models import naive_bayes, knn, svm_model, logistic, neural_network
import joblib

# --- Data Preparation ---
X_train, X_test, y_train, y_test = extract_features()

# oversample minority class (spam) so the model doesn't just predict ham every time
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# scale features for models that need it (knn, svm, neural network)
# MaxAbsScaler keeps sparse matrix format unlike StandardScaler
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_resampled_scaled = scaler.fit_transform(X_resampled)
X_test_scaled_resample = scaler.transform(X_test)

# --- Model Registry ---
# each module has: create_model(), get_param_grid(), NEEDS_SCALING
# to add a new model: create the file in models/, add it here
models = [
    ("Naive Bayes", naive_bayes),
    ("KNN", knn),
    ("SVM", svm_model),
    ("Logistic Regression", logistic),
    ("Neural Network", neural_network),
]

# --- Train, Evaluate & Tune All Models ---
all_results = {}
best_models = {}  # store tuned models for roc curve plotting

for name, module in models:
    model = module.create_model()
    param_grid = module.get_param_grid()

    # pick scaled or unscaled data based on what the model needs
    if module.NEEDS_SCALING:
        X_tr, X_te = X_train_scaled, X_test_scaled
        X_tr_sm, X_te_sm = X_resampled_scaled, X_test_scaled_resample
    else:
        X_tr, X_te = X_train, X_test
        X_tr_sm, X_te_sm = X_resampled, X_test

    # before smote - baseline to see how model does with imbalanced data
    evaluate_model(model, X_tr, X_te, y_train, y_test, f"{name} (Before SMOTE)")

    # after smote - should improve recall on spam (minority class)
    evaluate_model(module.create_model(), X_tr_sm, X_te_sm, y_resampled, y_test, f"{name} (After SMOTE)")

    # grid search to find best hyperparameters
    best_model, results = tune_and_evaluate(
        module.create_model(), param_grid, X_tr_sm, X_te_sm, y_resampled, y_test, name
    )
    all_results[name] = results
    best_models[name] = (best_model, X_te_sm)

# --- Comparison Table ---
# final side-by-side comparison of all tuned models
print("\n" + "=" * 70)
print("  MODEL COMPARISON (Tuned with GridSearchCV + SMOTE)")
print("=" * 70)
print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
print("-" * 70)
for name, res in all_results.items():
    auc_val = f"{res['auc']:.4f}" if 'auc' in res else "N/A"
    print(f"  {name:<25} {res['accuracy']:>10.4f} {res['precision']:>10.4f} {res['recall']:>10.4f} {res['f1']:>10.4f} {auc_val:>10}")
print("=" * 70)

# --- ROC Curve Plot ---
# shows how each model trades off between true positive rate and false positive rate
plt.figure(figsize=(10, 7))

for name, (best_mdl, X_te) in best_models.items():
    # get probability scores for roc curve
    if hasattr(best_mdl, 'predict_proba'):
        y_prob = best_mdl.predict_proba(X_te)[:, 1]
    elif hasattr(best_mdl, 'decision_function'):
        y_prob = best_mdl.decision_function(X_te)
    else:
        continue

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")

# diagonal line = random classifier (50/50 guess)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - All Models Comparison')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# --- Save Best Model ---
best_model_name = max(all_results, key=lambda model_name: all_results[model_name]['f1'])
print(f"\nBest model: {best_model_name} (F1: {all_results[best_model_name]['f1']:.4f})")

# save the trained model object and the vectorizer for backend inference
save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
os.makedirs(save_dir, exist_ok=True)

# best_models[name] is a tuple of (model_object, test_data) — we only need the model
joblib.dump(best_models[best_model_name][0], os.path.join(save_dir, 'best_model.pkl'))
joblib.dump(vectorizer, os.path.join(save_dir, 'vectorizer.pkl'))

print(f"Model saved to ml/saved_models/best_model.pkl")
print(f"Vectorizer saved to ml/saved_models/vectorizer.pkl")