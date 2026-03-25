from sklearn.naive_bayes import MultinomialNB


def create_model():
    # multinomial works best with tf-idf/count vectors (text data)
    return MultinomialNB()


def get_param_grid():
    # alpha = smoothing parameter, prevents zero probability issues
    return {
        'alpha': [0.1, 0.5, 1.0, 2.0]
    }


# naive bayes works fine without scaling since it uses probabilities
NEEDS_SCALING = False
