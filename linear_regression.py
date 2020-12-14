import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
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
    # filled_df.drop(["age", "education.num", "hours.per.week", "education", "fnlwgt", "capital.gain",
    #                 "capital.loss"], axis=1, inplace=True)

    # Since 90% of native.country is United-States, set all the other values to Other
    filled_df.loc[filled_df["native.country"] != "United-States", "native.country"] = "Other"

    # Save information about the original headers
    original_headers = list(filled_df)
    k = len(original_headers)
    # Convert the unordered categorical data into numerical data using one-hot encoding
    filled_df = pd.get_dummies(filled_df)

    # Remove any outliers (over 3 away from the standard deviation)
    filled_df = filled_df[(np.abs(filled_df - filled_df.mean()) <= (3 * filled_df.std())).all(axis=1)]

    # Convert the new dataframe into a numpy array
    data = filled_df.to_numpy()
    m = data.shape[0]
    true_data = np.copy(data)  # prevent changes on data from affecting true_data as well

    # Augment the data set with flags indicating whether a feature is missing or not
    new_headers = list(filled_df)
    k2 = len(new_headers)
    # Set a flag for each feature with a 20% chance of being missing (0)
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
    x = np.empty((m, k2 * 2 + 1), dtype=np.int64)
    # 0::2 = 0, 2, 4, ...; 1::2 = 1, 3, 5, ...
    x[:, 0] = 1
    x[:, 1::2] = data
    x[:, 2::2] = missing_features
    return x, true_data


def basic_completion(x, y):
    """
    Fill in the missing values based on the mean or mode of each feature and calculate the error

    :param x: the input matrix (m, 2k + 1)
    :param y: the true output (m, k)
    """
    mean = np.mean(x, axis=0)  # get the mean for each column
    mode = stats.mode(x)[0][0]  # get the mode for each column
    m, k2 = x.shape
    k = (k2 - 1) // 2
    y_pred = np.empty((m, k))

    for r in range(m):
        # Skip the feature flags
        for c in range(k):
            # Fill in missing features with the mode of each column
            cx = c * 2 + 1  # the column index for x and mode

            if x[r][cx + 1] == 0:
                # Use the mean if the values are real or the mode if the values are discrete
                if c < 6:
                    y_pred[r][c] = mean[cx]  # columns 0-5 are real
                else:
                    y_pred[r][c] = mode[cx]
            else:
                y_pred[r][c] = x[r][cx]

    error = np.sum(np.linalg.norm(y_pred - y) ** 2) / y.size
    print(f"Basic completion error = {error}")


def ridge_regression(x_train, y_train, k, el):
    """
    Calculate the optimal weight matrix using ridge regression

    :param x_train: the training input
    :param y_train: the training output
    :param k: the number of features in the output
    :param el: the regularization constant
    :return: the weight matrix that minimizes the error
    """
    return y_train.T @ x_train @ np.linalg.inv(x_train.T @ x_train + el * np.identity(k * 2 + 1))


def gd_ridge_regression(x_train, y_train, k, el, rng):
    """
    Calculate the optimal weight matrix using gradient descent

    :param x_train: the training input
    :param y_train: the training output
    :param k: the number of features in the output
    :param el: the regularization constant
    :param rng: the random number generator (np.random.default_rng())
    :return: the weight matrix that minimizes the error
    """
    a = 1e-15  # make alpha a small constant
    w_hat = rng.random((k, 2 * k + 1))  # the initial guess of W
    epsilon = 1e-2  # threshold between Wt+1 and Wt before declaring convergence

    while True:
        wp_hat = w_hat @ (np.identity(2 * k + 1) - a * x_train.T @ x_train - a * el * np.identity(2 * k + 1)) \
            + a * y_train.T @ x_train

        if np.linalg.norm(wp_hat - w_hat) < epsilon:
            break
        else:
            w_hat = wp_hat

    return w_hat


def lasso_regression(x_train, y_train, k, el, rng):
    """
    Calculate the optimal weight matrix using lasso regression

    :param x_train: the training input
    :param y_train: the training output
    :param k: the number of features in the output
    :param el: the regularization constant
    :param rng: the random number generator (np.random.default_rng())
    :return: the weight matrix that minimizes the error
    """
    w_hat = rng.random((k, 2 * k + 1))  # the initial guess of W
    epsilon = 1e-2  # threshold between Wt+1 and Wt before declaring convergence

    while True:
        wp_hat = np.copy(w_hat)
        # Choose a random coordinate j is in [0, 2k - 1] and update the entire column
        j = rng.integers(2 * k + 1)

        # Calculate the lower and upper bounds for Wij
        wi = w_hat
        xj = x_train[:, j]
        yi = y_train[:]
        lower = (-xj.T @ (yi - x_train @ wi.T) - el / 2) / (xj.T @ xj)
        upper = (-xj.T @ (yi - x_train @ wi.T) + el / 2) / (xj.T @ xj)

        # Update Wij based on the bounds and check for convergence
        for i in range(k):
            if w_hat[i][j] > upper[i]:
                wp_hat[i][j] = w_hat[i][j] - upper[i]
            elif w_hat[i][j] < lower[i]:
                wp_hat[i][j] = w_hat[i][j] - lower[i]
            else:
                wp_hat[i][j] = 0

        if np.linalg.norm(wp_hat - w_hat) < epsilon:
            break
        else:
            w_hat = wp_hat

    return w_hat


def get_error(w, x, y):
    """
    Calculate the error given either training or testing data

    :param w: the weight matrix (k, 2k + 1)
    :param x: the input (m, 2k + 1)
    :param y: the output (m, k)
    :return: the error (||XW^T - Y||^2/(m*k))
    """
    return np.sum(np.linalg.norm(x @ w.T - y) ** 2) / y.size


def main():
    # Read the CSV file and preprocess the data
    data_frame = pd.read_csv("adult.csv", encoding="utf-8", header=0)
    x, y = preprocess_data(data_frame)
    m, k = y.shape  # x has shape (m, 2k)
    basic_completion(x, y)  # find the ideal error to achieve

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

    # Compare the execution of each form of linear regression
    models = np.arange(4)
    errs_train = np.empty(len(models))
    errs_test = np.empty(len(models))

    start = time()
    w_hat = ridge_regression(x_train, y_train, k, 0.25)
    print(f"Regular ridge regression took {time() - start} s")
    y_pred = x_train @ w_hat.T
    errs_train[0] = get_error(w_hat, x_train, y_train)
    errs_test[0] = get_error(w_hat, x_test, y_test)
    print(f"err_train_rr = {errs_train[0]}")
    print(f"err_test_rr = {errs_test[0]}")

    start = time()
    w_hat = gd_ridge_regression(x_train, y_train, k, 0.25, rng)
    print(f"Gradient descent took {time() - start} s")
    errs_train[1] = get_error(w_hat, x_train, y_train)
    errs_test[1] = get_error(w_hat, x_test, y_test)
    print(f"err_train_gd = {errs_train[1]}")
    print(f"err_test_gd = {errs_test[1]}")

    start = time()
    w_hat = lasso_regression(x_train, y_train, k, 0.25, rng)
    print(f"Lasso regression took {time() - start} s")
    errs_train[2] = get_error(w_hat, x_train, y_train)
    errs_test[2] = get_error(w_hat, x_test, y_test)
    print(f"err_train_lr = {errs_train[2]}")
    print(f"err_test_lr = {errs_test[2]}")

    # Calculate the error in the worst case scenario (just guessing the weights)
    start = time()
    w_hat = rng.random((k, 2 * k + 1))
    print(f"Random guess took {time() - start} s")
    errs_train[3] = get_error(w_hat, x_train, y_train)
    errs_test[3] = get_error(w_hat, x_test, y_test)
    print(f"err_train_guess = {errs_train[3]}")
    print(f"err_test_guess = {errs_test[3]}")

    # Create a bar graph of the errors from different models
    plt.bar(models, errs_train, width=0.25)
    plt.bar(models + 0.25, errs_test, width=0.25)
    plt.title("Errors of Various Linear Regression Models")
    plt.xlabel("Type of Model")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.xticks(models + 0.125,
               ["Ridge Regression", "Gradient Descent", "Lasso Regression", "Random Guess"])
    plt.legend(["Training Error", "Testing Error"])
    plt.show()

    # Plot how the training and testing error change with lambda
    num_points = 10
    els = np.linspace(0.01, 1, num_points)  # lambda = 0 will result in a singular matrix
    errs_train = np.empty(num_points)
    errs_test = np.empty(num_points)

    for i in range(num_points):
        print(f"Point {i} / {num_points}...")
        w_hat = ridge_regression(x_train, y_train, k, els[i])
        errs_train[i] = get_error(w_hat, x_train, y_train)
        errs_test[i] = get_error(w_hat, x_test, y_test)

    # Find the lambda that results in the smallest training and testing error
    min_index_train = np.argmin(errs_train)
    min_index_test = np.argmin(errs_test)
    print(f"The minimum training error ({errs_train[min_index_train]}) occurs at "
          f"\u03BB = {els[min_index_train]}")
    print(f"The minimum testing error ({errs_test[min_index_train]}) occurs at "
          f"\u03BB = {els[min_index_test]}")

    # Show both the training and testing error
    plt.plot(els, errs_train)
    plt.plot(els, errs_test)
    plt.title("\u03BB vs. Error (Ridge Regression)")
    plt.xlabel("Regularization Constant (\u03BB)")
    plt.ylabel("Error")
    plt.legend(["Training Error", "Testing Error"])
    plt.show()

    # Plot only the training error
    plt.plot(els, errs_train)
    plt.title("\u03BB vs. Training Error (Ridge Regression)")
    plt.xlabel("Regularization Constant (\u03BB)")
    plt.ylabel("Training Error")
    plt.show()

    # Plot only the testing error
    plt.plot(els, errs_test)
    plt.title("\u03BB vs. Testing Error (Ridge Regression)")
    plt.xlabel("Regularization Constant (\u03BB)")
    plt.ylabel("Testing Error")
    plt.show()

    # Plot gradient descent ridge regression
    num_points = 10
    els = np.linspace(0.1, 1, num_points)  # lambda = 0 will result in a singular matrix
    errs_train = np.empty(num_points)
    errs_test = np.empty(num_points)

    for i in range(num_points):
        print(f"Point {i} / {num_points}...")
        w_hat = gd_ridge_regression(x_train, y_train, k, els[i], rng)
        errs_train[i] = get_error(w_hat, x_train, y_train)
        errs_test[i] = get_error(w_hat, x_test, y_test)

    # Find the lambda that results in the smallest training and testing error
    min_index_train = np.argmin(errs_train)
    min_index_test = np.argmin(errs_test)
    print(f"The minimum training error ({errs_train[min_index_train]}) occurs at "
          f"\u03BB = {els[min_index_train]}")
    print(f"The minimum testing error ({errs_test[min_index_train]}) occurs at "
          f"\u03BB = {els[min_index_test]}")

    plt.plot(els, errs_train)
    plt.plot(els, errs_test)
    plt.title("\u03BB vs. Error (Gradient Descent Ridge Regression)")
    plt.xlabel("Regularization Constant (\u03BB)")
    plt.ylabel("Error")
    plt.legend(["Training Error", "Testing Error"])
    plt.show()

    plt.plot(els, errs_train)
    plt.title("\u03BB vs. Training Error (Gradient Descent Ridge Regression)")
    plt.xlabel("Regularization Constant (\u03BB)")
    plt.ylabel("Training Error")
    plt.show()

    plt.plot(els, errs_test)
    plt.title("\u03BB vs. Testing Error (Gradient Descent Ridge Regression)")
    plt.xlabel("Regularization Constant (\u03BB)")
    plt.ylabel("Testing Error")
    plt.show()

    # Plot lasso regression
    num_points = 10
    els = np.linspace(0.01, 1, num_points)  # lambda = 0 will result in a singular matrix
    errs_train = np.empty(num_points)
    errs_test = np.empty(num_points)

    for i in range(num_points):
        print(f"Point {i} / {num_points}...")
        w_hat = lasso_regression(x_train, y_train, k, els[i], rng)
        errs_train[i] = get_error(w_hat, x_train, y_train)
        errs_test[i] = get_error(w_hat, x_test, y_test)

    # Find the lambda that results in the smallest training and testing error
    min_index_train = np.argmin(errs_train)
    min_index_test = np.argmin(errs_test)
    print(f"The minimum training error ({errs_train[min_index_train]}) occurs at "
          f"\u03BB = {els[min_index_train]}")
    print(f"The minimum testing error ({errs_test[min_index_train]}) occurs at "
          f"\u03BB = {els[min_index_test]}")

    plt.plot(els, errs_train)
    plt.plot(els, errs_test)
    plt.title("\u03BB vs. Error (Lasso Regression)")
    plt.xlabel("Regularization Constant (\u03BB)")
    plt.ylabel("Error")
    plt.legend(["Training Error", "Testing Error"])
    plt.show()

    plt.plot(els, errs_train)
    plt.title("\u03BB vs. Training Error (Lasso Regression)")
    plt.xlabel("Regularization Constant (\u03BB)")
    plt.ylabel("Training Error")
    plt.show()

    plt.plot(els, errs_test)
    plt.title("\u03BB vs. Testing Error (Lasso Regression)")
    plt.xlabel("Regularization Constant (\u03BB)")
    plt.ylabel("Testing Error")
    plt.show()


if __name__ == "__main__":
    main()
