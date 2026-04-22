from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from data_loader import load_data
from text_processing import preprocess_text

# initialize vectorizer at module level so it can be reused later for inference
vectorizer = TfidfVectorizer()


def extract_features():
    # load raw data and run preprocessing pipeline
    df = preprocess_text(load_data())
    X = df['message']
    y = df['label']

    # 80/20 split, random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # fit on train only, transform both - avoids data leakage
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test
