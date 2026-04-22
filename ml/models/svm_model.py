from sklearn.svm import SVC


def create_model():
    # probability=True needed to get predict_proba for auc calculation
    return SVC(probability=True)


def get_param_grid():
    # linear kernel ignores gamma, so only tuning C
    return {
        'C': [1, 10],
        'kernel': ['linear']
    }


# svm is sensitive to feature scale, needs normalization
NEEDS_SCALING = True
