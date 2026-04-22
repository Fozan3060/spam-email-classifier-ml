from sklearn.neighbors import KNeighborsClassifier


def create_model():
    return KNeighborsClassifier(n_neighbors=5)


def get_param_grid():
    return {
        'n_neighbors': [3, 5, 7, 9],       # how many neighbors to check
        'weights': ['uniform', 'distance']  # uniform = equal vote, distance = closer = more weight
    }


# knn uses distance calculations so features must be on the same scale
NEEDS_SCALING = True
