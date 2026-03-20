from data_loader import load_data
from eda import perform_eda
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from text_processing import text_processing

vectorizer = TfidfVectorizer()


def feature_extraction():
    df=text_processing(perform_eda(load_data()))
    x= df['message']
    y= df["label"]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,shuffle=True)
    X_train_tfidf = vectorizer.fit_transform(x_train)
    X_test_tfidf = vectorizer.transform(x_test)
    print(X_train_tfidf.shape, X_test_tfidf.shape)
    return X_train_tfidf, X_test_tfidf, y_train, y_test


feature_extraction()


