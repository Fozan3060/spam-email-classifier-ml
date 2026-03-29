import joblib
import os
import sys

# add ml/ to path so we can import text_processing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml'))

from text_processing import preprocess_text
import pandas as pd

# load everything once when this file is imported — not per request
saved_dir = os.path.join(os.path.dirname(__file__), '..', 'ml', 'saved_models')

model = joblib.load(os.path.join(saved_dir, 'best_model.pkl'))
vectorizer = joblib.load(os.path.join(saved_dir, 'vectorizer.pkl'))
scaler = joblib.load(os.path.join(saved_dir, 'scaler.pkl'))
needs_scaling = joblib.load(os.path.join(saved_dir, 'needs_scaling.pkl'))


def predict_spam(message: str):
    # step 1: put the message in a dataframe (same format as training)
    df = pd.DataFrame({'message': [message]})

    # step 2: preprocess — lowercase, remove special chars, stopwords, lemmatize
    df = preprocess_text(df)
    processed_text = df['message'].iloc[0]

    # step 3: convert to tfidf vector using the saved vectorizer
    text_vector = vectorizer.transform([processed_text])

    # step 4: scale if the best model needs it (knn, svm, neural network)
    if needs_scaling:
        text_vector = scaler.transform(text_vector)

    # step 5: predict using the saved model
    prediction = model.predict(text_vector)[0]

    # step 6: get probability if available
    probability = None
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(text_vector)[0][1]  # spam probability

    return {
        'prediction': 'spam' if prediction == 1 else 'ham',
        'is_spam': bool(prediction),
        'confidence': round(float(probability), 4) if probability is not None else None,
        'processed_text': processed_text
    }
