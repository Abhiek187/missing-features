import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from time import time


def preprocess_data(data_frame):
    """
    Convert the data set into a numerical data set with 2k features

    :param data_frame: the original data set as a pandas dataframe
    :return: the preprocessed input data set as a numpy array and the regular input as the output data set
    """
    # Prune any rows with missing data points (indicated by ?)
    filled_df = data_frame[~(data_frame == "?").any(axis=1)]

    # Prune the education column since education.num makes it obsolete
    del filled_df["education"]

    # Since 90% of native.country is United-States, set all the other values to Other
    filled_df.loc[filled_df["native.country"] != "United-States", "native.country"] = "Other"

    # Convert the unordered categorical data into numerical data using one-hot encoding
    # Save information about the original headers
    original_headers = list(filled_df)
    k = len(original_headers)
    filled_df = pd.get_dummies(filled_df)
    # Convert the new dataframe into a numpy array
    data = filled_df.to_numpy()
    m = data.shape[0]
    true_data = np.copy(data)  # prevent changes on data from affecting true_data as well

    # Augment the data set with flags indicating whether a feature is missing or not
    new_headers = list(filled_df)
    k2 = len(new_headers)
    # Set a flag for each feature with an 80% chance of being missing (0)
    rand_ints = np.random.randint(10, size=(m, k))
    flags = (rand_ints > 1).astype(int)
    missing_features = np.zeros((m, k2), dtype=np.int64)

    # Match the flags from the original header order to the new header order
    for r in range(m):
        # Create a dictionary mapping each feature to its missing flag
        features_dict = dict(zip(original_headers, flags[r]))
        for c in range(k2):
            col = new_headers[c].split("_")[0]  # one-hot encodings start with header name_value
            missing_features[r][c] = features_dict[col]

            if missing_features[r][c] == 0:
                data[r][c] = 0  # replace missing features with 0

    # Interleave the data values with their missing flags (feature1, present?, feature2, present?, ...)
    x = np.empty((m, k2 * 2), dtype=np.int64)
    # 0::2 = 0, 2, 4, ...; 1::2 = 1, 3, 5, ...
    x[:, 0::2] = data
    x[:, 1::2] = missing_features
    return x, true_data


def main():
    # Read the CSV file and preprocess the data
    data_frame = pd.read_csv("adult.csv", encoding="utf-8", header=0)
    x, y = preprocess_data(data_frame)
    m, k = y.shape  # x has shape (m, 2k)

    # Shuffle x and y
    rng = np.random.default_rng()
    rand_order = rng.permutation(m)
    x = x[rand_order]
    y = y[rand_order]

    # Set aside 75% training data and 25% testing data
    split_index = round(m * 0.75)  # the index must be an int
    x_train = x[:split_index]
    x_test = x[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    # Score (LinearRegression): 0.8145993658516727
    # LinearRegression took 0.34897446632385254 s
    classifier = LinearRegression(normalize=True, n_jobs=-1)
    start = time()
    # params = {
    #     "fit_intercept": [True, False],
    #     "normalize": [False, True]
    # }
    # classifier = GridSearchCV(classifier, params, n_jobs=-1).fit(x_train, y_train)
    # print(f"Best estimator: {classifier.best_estimator_}")
    # print(f"Best score: {classifier.best_score_}")
    classifier.fit(x_train, y_train)
    # print(f"weights: {classifier.coef_}")
    # print(f"bias: {classifier.intercept_}")
    # print(f"Prediction: {classifier.predict(x_test)}")
    # print(f"Actual: {y_test}")
    print(f"Score (LinearRegression): {classifier.score(x_test, y_test)}")
    print(f"LinearRegression took {time() - start} s")

    # Score (Ridge): 0.8140767462789897
    # Ridge took 0.09673619270324707 s
    classifier = Ridge()
    start = time()
    # params = {
    #     "alpha": [1e-2, 1, 1e2],
    #     "fit_intercept": [True, False],
    #     "normalize": [False, True]
    # }
    # classifier = GridSearchCV(classifier, params, n_jobs=-1).fit(x_train, y_train)
    # print(f"Best estimator: {classifier.best_estimator_}")
    # print(f"Best score: {classifier.best_score_}")
    classifier.fit(x_train, y_train)
    # print(f"weights: {classifier.coef_}")
    # print(f"bias: {classifier.intercept_}")
    # print(f"Prediction: {classifier.predict(x_test)}")
    # print(f"Actual: {y_test}")
    print(f"Score (Ridge): {classifier.score(x_test, y_test)}")
    print(f"Ridge took {time() - start} s")

    # Score (Lasso): 0.8133384974972955
    # Lasso took 0.140655517578125 s
    classifier = Lasso(alpha=1e-6, precompute=True, tol=0.1)
    start = time()
    # params = {
    #     # "alpha": [1e-7, 1e-6, 1e-5],
    #     # "fit_intercept": [True, False],
    #     # "normalize": [False, True],
    #     # "precompute": [False, True],
    #     # "warm_start": [False, True],
    #     # "positive": [False, True],
    #     # "selection": ["cyclic", "random"]
    # }
    # classifier = GridSearchCV(classifier, params, n_jobs=-1).fit(x_train, y_train)
    # print(f"Best estimator: {classifier.best_estimator_}")
    # print(f"Best score: {classifier.best_score_}")
    classifier.fit(x_train, y_train)
    # # print(f"weights: {classifier.coef_}")
    # # print(f"bias: {classifier.intercept_}")
    # # print(f"Prediction: {classifier.predict(x_test)}")
    # # print(f"Actual: {y_test}")
    print(f"Score (Lasso): {classifier.score(x_test, y_test)}")
    print(f"Lasso took {time() - start} s")


if __name__ == "__main__":
    main()
