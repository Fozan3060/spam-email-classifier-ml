from sklearn.linear_model import LogisticRegression


def create_model():
    # max_iter=1000 because default 100 may not converge on this dataset
    return LogisticRegression(max_iter=1000, solver='liblinear')


def get_param_grid():
    # using l1_ratio instead of penalty (penalty is deprecated in sklearn 1.8+)
    # l1_ratio=1 means l1, l1_ratio=0 means l2
    return {
        'C': [0.01, 0.1, 1, 10, 100],
        'l1_ratio': [0, 0.5, 1]
    }


# logistic regression works ok without scaling on tf-idf (already 0-1 range)
NEEDS_SCALING = False
