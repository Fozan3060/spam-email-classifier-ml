from feature_extraction import feature_extraction
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics , svm
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.neural_network import MLPClassifier

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
X_train_scaled = scaler.fit_transform(X_train_tfidf)
X_test_scaled = scaler.transform(X_test_tfidf)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))

print("KNN after oversampling")
X_resampled_scaled = scaler.fit_transform(X_resampled_train)
X_test_scaled = scaler.transform(X_test_tfidf)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_resampled_scaled, y_resampled_train)
y_pred = knn.predict(X_test_scaled)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))

print("Finding best K value using k cross validation")
parameters = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
clf = GridSearchCV(KNeighborsClassifier(), parameters, scoring='f1', cv=5)
clf.fit(X_resampled_scaled, y_resampled_train)

print("Best Parameters found KNN:", clf.best_params_)
print("Best Cross-Validation Score (F1): KNN", clf.best_score_)
# Final Evaluation of the tuned model
y_pred_tuned = clf.predict(X_test_scaled)
print("Final Tuned KNN Accuracy:", metrics.accuracy_score(y_test, y_pred_tuned))
print("Final Tuned KNN Classification Report:\n", metrics.classification_report(y_test, y_pred_tuned))


print("SVM after oversampling")
svc = svm.SVC()
param_grid = {                                                                                                                             
      'C': [1, 10],                                                                                                                          
      'gamma': [0.1, 0.01],                                                                                                                  
      'kernel': ['linear']
  }   
grid_search_svm = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='f1')
grid_search_svm.fit(X_resampled_scaled, y_resampled_train)
print("Best hyperparameters found in svm: ", grid_search_svm.best_params_)
print("Best Cross-Validation Score (F1) SVM:", grid_search_svm.best_score_)
# Final Evaluation of the tuned model
y_pred_tuned = grid_search_svm.predict(X_test_scaled)
print("Final Tuned SVM Accuracy:", metrics.accuracy_score(y_test, y_pred_tuned))
print("Final Tuned SVM Classification Report:\n", metrics.classification_report(y_test, y_pred_tuned))

print("Logistic Regression")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print("Classificaiton Report using Logisctic Regression : ",metrics.classification_report(y_test,y_pred))

print("Logistic Regression after oversampling")
model = LogisticRegression(max_iter=1000)
model.fit(X_resampled_train, y_resampled_train)
y_pred = model.predict(X_test_tfidf)
print("Classificaiton Report using Logisctic Regression after oversampling: ",metrics.classification_report(y_test,y_pred))

print("Logictic Regression Best hyper parameters using grid serach ")
param_grid = {
    'C': np.logspace(-3, 3, 7), # Generates 7 values on a logarithmic scale
    'penalty': ['l1', 'l2']
}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,            # Use 5-fold cross-validation
    scoring='f1', # Metric to optimize
    verbose=1,       # Output progress
    n_jobs=-1        # Use all available processors
)
grid_search.fit(X_resampled_train, y_resampled_train)
print("Tuned Hyperparameters (best parameters):", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)
grid_search.predict(X_test_tfidf)

print("Neural Networking:")
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred=model.predict(X_test_scaled)
print("Score",metrics.classification_report(y_test,y_pred))

print("Neural networking after oversampling")
model.fit(X_resampled_scaled, y_resampled_train)
y_pred=model.predict(X_test_scaled)
print("Score",metrics.classification_report(y_test,y_pred))


print("Neural netowrking after grid search cv")
param_grid = {
    'hidden_layer_sizes': [(50,), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.01],
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1) #
grid_search.fit(X_resampled_scaled, y_resampled_train)
y_pred=grid_search.predict(X_test_scaled)
print("Score",metrics.classification_report(y_test,y_pred))