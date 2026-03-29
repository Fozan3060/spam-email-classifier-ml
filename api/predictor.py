import joblib
import os
import sys

# add ml/ to path so we can import text_processing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml'))

from text_processing import preprocess_text
import pandas as pd

# load the saved model and vectorizer once when this file is imported
# not inside the function — loading every request would be slow
model_path = os.path.join(os.path.dirname(__file__), '..', 'ml', 'saved_models', 'best_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'ml', 'saved_models', 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


def predict_spam(message: str):
    # step 1: put the message in a dataframe (same format as training)
    df = pd.DataFrame({'message': [message]})

    # step 2: preprocess — lowercase, remove special chars, stopwords, lemmatize
    df = preprocess_text(df)
    processed_text = df['message'].iloc[0]

    # step 3: convert to tfidf vector using the saved vectorizer
    text_vector = vectorizer.transform([processed_text])

    # step 4: predict using the saved model
    prediction = model.predict(text_vector)[0]

    # step 5: get probability if available
    probability = None
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(text_vector)[0][1]  # spam probability

    return {
        'prediction': 'spam' if prediction == 1 else 'ham',
        'is_spam': bool(prediction),
        'confidence': round(float(probability), 4) if probability is not None else None,
        'processed_text': processed_text
    }
