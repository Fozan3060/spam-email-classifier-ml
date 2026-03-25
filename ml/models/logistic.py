from sklearn.linear_model import LogisticRegression


def create_model():
    # liblinear solver supports both l1 and l2 penalty
    # max_iter=1000 because default 100 may not converge on this dataset
    return LogisticRegression(max_iter=1000, solver='liblinear')


def get_param_grid():
    return {
        'C': [0.01, 0.1, 1, 10, 100],  # regularization strength
        'penalty': ['l1', 'l2']          # l1 can zero out useless features, l2 shrinks all
    }


# logistic regression works ok without scaling on tf-idf (already 0-1 range)
NEEDS_SCALING = False
