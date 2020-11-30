# Just testing to see if I still remember sklearn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from time import time

features_list = ["age", "workclass", "education.num", "marital.status", "race", "sex"] # variables to help infer who survived
labels_list = ["income"] # who survived?

# Get the features and label columns from the training csv
data_frame = pd.read_csv("adult.csv", encoding="utf-8", header=0)
features = data_frame[features_list]
labels = data_frame[labels_list]
# Convert categorical data into numerical data
features = pd.get_dummies(features)
# Fill in missing values with some number
imputer = SimpleImputer()
features = imputer.fit_transform(features)

# Split the dataset into training and testing data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Choose a classifier, fit, and predict model
# Score (GaussianNB): 0.775104923738356
# GaussianNB took 0.08158326148986816 s
classifier = GaussianNB(var_smoothing=1e-3)
start = time()
"""params = {
        "var_smoothing": [1e-7, 1e-5, 1e-3, 1e-1, 1, 10]
}
classifier = GridSearchCV(classifier, params, n_jobs=-1).fit(features_train, labels_train.to_numpy().ravel())
print(f"Best estimator: {classifier.best_estimator_}")
print(f"Best score: {classifier.best_score_}")"""
classifier.fit(features_train, labels_train.to_numpy().ravel())
#predictions = classifier.predict(features_test)
#print(predictions)
print(f"Score (GaussianNB): {classifier.score(features_test, labels_test)}")
print(f"GaussianNB took {time() - start} s")

# Score (SVC): 0.8250588596581021
# SVC took 1025.2103826999664 s
classifier = SVC(C=100000, coef0=5, degree=5, gamma="scale", kernel="poly", tol=1)
start = time()
"""params = {
	"C": [1e-2, 1, 100, 10000]
}
classifier = GridSearchCV(classifier, params, n_jobs=-1).fit(features_train, labels_train.to_numpy().ravel())
print(f"Best estimator: {classifier.best_estimator_}")
print(f"Best score: {classifier.best_score_}")"""
classifier.fit(features_train, labels_train.to_numpy().ravel())
#predictions = classifier.predict(features_test)
#print(predictions)
print(f"Score (SVC): {classifier.score(features_test, labels_test)}")
print(f"SVC took {time() - start} s")

# Score (DecisionTreeClassifier): 0.8286416214556249
# DecisionTreeClassifier took 0.09577679634094238 s
classifier = DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_split=500)
start = time()
"""params = {
	"min_samples_split": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
}
classifier = GridSearchCV(classifier, params, n_jobs=-1).fit(features_train, labels_train.to_numpy().ravel())
print(f"Best estimator: {classifier.best_estimator_}")
print(f"Best score: {classifier.best_score_}")"""
classifier.fit(features_train, labels_train.to_numpy().ravel())
#predictions = classifier.predict(features_test)
#print(predictions)
print(f"Score (DecisionTreeClassifier): {classifier.score(features_test, labels_test)}")
print(f"DecisionTreeClassifier took {time() - start} s")

# Score (KNeighborsClassifier): 0.8198382638959976
# KNeighborsClassifier took 15.495917558670044 s
classifier = KNeighborsClassifier(n_neighbors=30, algorithm="brute", leaf_size=1, n_jobs=-1)
start = time()
"""params = {
	"n_neighbors": [30, 40, 50]
}
classifier = GridSearchCV(classifier, params, n_jobs=-1).fit(features_train, labels_train.to_numpy().ravel())
print(f"Best estimator: {classifier.best_estimator_}")
print(f"Best score: {classifier.best_score_}")"""
classifier.fit(features_train, labels_train.to_numpy().ravel())
#predictions = classifier.predict(features_test)
#print(predictions)
print(f"Score (KNeighborsClassifier): {classifier.score(features_test, labels_test)}")
print(f"KNeighborsClassifier took {time() - start} s")

# Test the model on the test csv file
"""test_df = pd.read_csv("/kaggle/input/titanic/test.csv", encoding="utf-8", header=0)
features_test = imputer.fit_transform(pd.get_dummies(test_df[features_list]))"""

# Get predicted values, then store in a new csv file
"""classifier = GaussianNB()
classifier.fit(features, labels.values.ravel())
preds = classifier.predict(features_test)

new_df = pd.DataFrame({"PassengerId": list(range(892, 1310)), "Survived": preds})
new_df.to_csv("/kaggle/working/my_submission.csv", index=False)"""
