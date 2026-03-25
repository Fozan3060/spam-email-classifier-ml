from sklearn.neural_network import MLPClassifier


def create_model():
    # single hidden layer with 100 neurons, enough for this dataset size
    # max_iter=1000 to give it enough epochs to converge
    return MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)


def get_param_grid():
    # reduced grid - NN grid search is slow (each fit trains a full network)
    # adam is almost always better than sgd, so dropped sgd
    return {
        'hidden_layer_sizes': [(50,), (100,)],  # number of neurons per layer
        'activation': ['relu'],                  # relu is standard, tanh rarely wins
        'alpha': [0.0001, 0.01]                  # regularization to prevent overfitting
    }


# neural networks are very sensitive to feature scale
NEEDS_SCALING = True
