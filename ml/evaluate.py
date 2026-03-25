from sklearn.model_selection import GridSearchCV
from metrics import compute_metrics, print_metrics


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # train and evaluate a single model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # try to get probability scores for auc calculation
    # predict_proba gives actual probabilities, decision_function gives confidence scores
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1 (spam)
    elif hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_test)

    results = compute_metrics(y_test, y_pred, y_prob)
    print_metrics(model_name, results)
    return results


def tune_and_evaluate(model, param_grid, X_train, X_test, y_train, y_test, model_name):
    # use grid search to find best hyperparameters via cross validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',  # optimizing for f1 because of class imbalance
        n_jobs=-1       # use all cpu cores for speed
    )
    grid_search.fit(X_train, y_train)

    print(f"\n  Best Parameters ({model_name}): {grid_search.best_params_}")
    print(f"  Best CV F1 Score: {grid_search.best_score_:.4f}")

    y_pred = grid_search.predict(X_test)

    # get probabilities from the best model found by grid search
    y_prob = None
    best_model = grid_search.best_estimator_
    if hasattr(best_model, 'predict_proba'):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    elif hasattr(best_model, 'decision_function'):
        y_prob = best_model.decision_function(X_test)

    results = compute_metrics(y_test, y_pred, y_prob)
    print_metrics(f"{model_name} (Tuned)", results)
    return grid_search.best_estimator_, results
