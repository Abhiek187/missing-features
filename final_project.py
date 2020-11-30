# Just testing to see if I still remember sklearn
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from time import time

features_list = ["age", "workclass", "education.num", "marital.status", "race",
                 "sex"]  # variables to help infer who survived
labels_list = ["income"]  # who survived?

# Get the features and label columns from the training csv
data_frame = pd.read_csv("adult.csv", encoding="utf-8", header=0)
features = data_frame[features_list]
labels = data_frame[labels_list]
# Convert categorical data into numerical data
features = pd.get_dummies(features)
labels = pd.get_dummies(labels, drop_first=True)  # make output 1 dimensional
# Fill in missing values with some number
imputer = SimpleImputer()
features = imputer.fit_transform(features)
labels = imputer.fit_transform(labels)

# Split the dataset into training and testing data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
                                                                            random_state=42)

# Score (DecisionTreeClassifier): 0.8286416214556249
# DecisionTreeClassifier took 0.09577679634094238 s
classifier = DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_split=500)
start = time()
# params = {
# 	"min_samples_split": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# }
# classifier = GridSearchCV(classifier, params, n_jobs=-1).fit(features_train, labels_train.ravel())
# print(f"Best estimator: {classifier.best_estimator_}")
# print(f"Best score: {classifier.best_score_}")
classifier.fit(features_train, labels_train.ravel())
# predictions = classifier.predict(features_test)
# print(predictions)
print(f"Score (DecisionTreeClassifier): {classifier.score(features_test, labels_test)}")
print(f"DecisionTreeClassifier took {time() - start} s")

# Score (Perceptron): 0.7055993448664142
# Perceptron took 0.08676719665527344 s
classifier = Perceptron(penalty="l1", alpha=0.01, eta0=10, n_jobs=-1)
start = time()
# params = {
#     "eta0": [1, 10, 100, 1000]
# }
# classifier = GridSearchCV(classifier, params, n_jobs=-1).fit(features_train, labels_train.ravel())
# print(f"Best estimator: {classifier.best_estimator_}")
# print(f"Best score: {classifier.best_score_}")
classifier.fit(features_train, labels_train.ravel())
print(f"Score (Perceptron): {classifier.score(features_test, labels_test)}")
print(f"Perceptron took {time() - start} s")

# Score (SVC): 0.8250588596581021
# SVC took 872.8663313388824 s
classifier = SVC(C=100000, coef0=5, degree=5, gamma="scale", kernel="poly", tol=1)
start = time()
# params = {
#     "C": [1e-2, 1, 100, 10000]
# }
# classifier = GridSearchCV(classifier, params, n_jobs=-1).fit(features_train, labels_train.ravel())
# print(f"Best estimator: {classifier.best_estimator_}")
# print(f"Best score: {classifier.best_score_}")
classifier.fit(features_train, labels_train.ravel())
# predictions = classifier.predict(features_test)
# print(predictions)
print(f"Score (SVC): {classifier.score(features_test, labels_test)}")
print(f"SVC took {time() - start} s")

# Score (LogisticRegression): 0.8208619101238612
# LogisticRegression took 11.388295412063599 s
classifier = LogisticRegression(fit_intercept=False, max_iter=1000, multi_class="multinomial", n_jobs=-1)
start = time()
# params = {
#     # "penalty": ["l1", "l2", "elasticnet", "none"],
#     "C": [1e-2, 1, 1e2],
#     "fit_intercept": [True, False],
#     # "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
#     "multi_class": ["auto", "ovr", "multinomial"],
#     "warm_start": [False, True]
# }
# classifier = GridSearchCV(classifier, params, n_jobs=-1).fit(features_train, labels_train.ravel())
# print(f"Best estimator: {classifier.best_estimator_}")
# print(f"Best score: {classifier.best_score_}")
classifier.fit(features_train, labels_train.ravel())
print(f"Score (LogisticRegression): {classifier.score(features_test, labels_test)}")
print(f"LogisticRegression took {time() - start} s")

# Score (LinearRegression): 0.29749878612984665
# LinearRegression took 0.03781867027282715 s
classifier = LinearRegression(fit_intercept=False, n_jobs=-1)
start = time()
# params = {
#     "fit_intercept": [True, False],
#     "normalize": [False, True]
# }
# classifier = GridSearchCV(classifier, params, n_jobs=-1).fit(features_train, labels_train.ravel())
# print(f"Best estimator: {classifier.best_estimator_}")
# print(f"Best score: {classifier.best_score_}")
classifier.fit(features_train, labels_train.ravel())
print(f"Score (LinearRegression): {classifier.score(features_test, labels_test)}")
print(f"LinearRegression took {time() - start} s")

# Test the model on the test csv file
"""test_df = pd.read_csv("/kaggle/input/titanic/test.csv", encoding="utf-8", header=0)
features_test = imputer.fit_transform(pd.get_dummies(test_df[features_list]))"""

# Get predicted values, then store in a new csv file
"""classifier = GaussianNB()
classifier.fit(features, labels.values.ravel())
preds = classifier.predict(features_test)

new_df = pd.DataFrame({"PassengerId": list(range(892, 1310)), "Survived": preds})
new_df.to_csv("/kaggle/working/my_submission.csv", index=False)"""
