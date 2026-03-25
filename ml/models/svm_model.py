from sklearn.svm import SVC


def create_model():
    # probability=True needed to get predict_proba for auc calculation
    return SVC(probability=True)


def get_param_grid():
    return {
        'C': [1, 10],             # regularization - higher = fits training data more
        'gamma': [0.1, 0.01],     # how far each training example's influence reaches
        'kernel': ['linear']      # linear works best for text (high dimensional data)
    }


# svm is sensitive to feature scale, needs normalization
NEEDS_SCALING = True
