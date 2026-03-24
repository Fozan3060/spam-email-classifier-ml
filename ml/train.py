from feature_extraction import feature_extraction
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from imblearn.over_sampling import SMOTE


X_train_tfidf, X_test_tfidf, y_train, y_test = feature_extraction()

sm = SMOTE(random_state=42)
gnb = MultinomialNB()
gnb.fit(X_train_tfidf, y_train)

y_pred = gnb.predict(X_test_tfidf)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))


print("After Oversampling of minority class")
X_resampled, y_resampled = sm.fit_resample(X_train_tfidf, y_train)
gnb.fit(X_resampled, y_resampled)
y_pred = gnb.predict(X_test_tfidf)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))



