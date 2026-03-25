from feature_extraction import feature_extraction
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import KNeighborsClassifier


X_train_tfidf, X_test_tfidf, y_train, y_test = feature_extraction()
sm = SMOTE(random_state=42)
print("After Oversampling of minority class")
X_resampled_train, y_resampled_train = sm.fit_resample(X_train_tfidf, y_train)
scaler = MaxAbsScaler()


print("Naive Bayes using mutlinomialNB")
gnb = MultinomialNB()
gnb.fit(X_train_tfidf, y_train)

y_pred = gnb.predict(X_test_tfidf)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))


print("Naive Bayes after oversampling")
gnb.fit(X_resampled_train, y_resampled_train)
y_pred = gnb.predict(X_test_tfidf)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))

print("KNN before oversampling")
scaler.fit_transform(X_train_tfidf)
X_test_scaled = scaler.transform(X_test_tfidf)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_tfidf, y_train)
y_pred = knn.predict(X_test_scaled)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))

print("KNN after oversampling")
scaler.fit_transform(X_resampled_train)
X_test_scaled = scaler.transform(X_test_tfidf)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_resampled_train, y_resampled_train)
y_pred = knn.predict(X_test_scaled)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))



