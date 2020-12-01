import numpy as np
import pandas as pd


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
    true_data = data

    # Augment the data set with flags indicating whether a feature is missing or not
    new_headers = list(filled_df)
    k2 = len(new_headers)
    # Set a flag for each feature with an 80% chance of being missing (0)
    rand_ints = np.random.randint(10, size=(m, k))
    flags = (rand_ints > 1).astype(int)
    missing_features = np.zeros((m, k2))

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
    x = np.empty((m, k2*2))
    # 0::2 = 0, 2, 4, ...; 1::2 = 1, 3, 5, ...
    x[:, 0::2] = data
    x[:, 1::2] = missing_features
    return x, true_data


def main():
    # Read the CSV file and preprocess the data
    data_frame = pd.read_csv("adult.csv", encoding="utf-8", header=0)
    x, y = preprocess_data(data_frame)
    print(f"x[:10]: {x[:10]}")
    print(f"y[:10]: {y[:10]}")


if __name__ == "__main__":
    main()
