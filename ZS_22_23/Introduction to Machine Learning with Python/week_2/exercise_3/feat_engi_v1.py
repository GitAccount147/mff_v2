#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
# diabetes/linnerud/wine
parser.add_argument("--dataset", default="wine", type=str, help="Standard sklearn dataset to load")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()
    data, target = dataset.data, dataset.target
    print(data.shape, target.shape)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, target)
    #print(X_train[0])
    #print(X_train[:][4])
    #print(X_train[0][4].is_integer())

    # TODO: Process the input columns in the following way:
    #
    # - if a column has only integer values, consider it a categorical column
    #   (days in a week, dog breed, ...; in general, integer values can also
    #   represent numerical non-categorical values, but we use this assumption
    #   for the sake of exercise). Encode the values with one-hot encoding
    #   using `sklearn.preprocessing.OneHotEncoder` (note that its output is by
    #   default sparse, you can use `sparse=False` to generate dense output;
    #   also use `handle_unknown="ignore"` to ignore missing values in test set).
    sample_size, feat_size = X_train.shape[0], X_train.shape[1]
    tran = np.transpose(X_train)
    int_cols = []
    real_cols = []
    #print(tran)
    for i in range(feat_size):
        all_int = True
        for j in range(sample_size):
            if not tran[i][j].is_integer():  # isinstance(tran[i][j], int)
                all_int = False
        if all_int:
            print("column is all_int:", i)
            int_cols.append(i)
        else:
            real_cols.append(i)
    print(int_cols)
    X_train_int = []
    X_train_real = []
    for i in range(feat_size):
        if i in int_cols:
            X_train_int.append(tran[i])
        else:
            X_train_real.append(tran[i])

    X_train_int = np.transpose(X_train_int)
    X_train_real = np.transpose(X_train_real)
    #print(X_train_int)
    #print(X_train_real)

    """
    enc2 = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')
    X2 = [['Male', 1], ['Female', 3], ['Female', 2]]
    enc2.fit(X2)
    print(enc2.categories_)
    print(enc2.transform([['Female', 1], ['Male', 4]]).toarray())
    """

    enc = sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore")
    enc.fit(X_train_int)
    X_train_int_new = enc.transform(X_train_int)
    #X_train_int_new = enc.fit_transform(X_train_int)
    #print(X_train_int_new[0])

    #
    # - for the rest of the columns, normalize their values so that they
    #   have mean 0 and variance 1; use `sklearn.preprocessing.StandardScaler`.
    #

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X_train_real)
    X_train_real_new = scaler.transform(X_train_real)
    #print(X_train_real[0])
    #print(X_train_real_new[0])

    # In the output, first there should be all the one-hot categorical features,
    # and then the real-valued features. To process different dataset columns
    # differently, you can use `sklearn.compose.ColumnTransformer`.

    tr = sklearn.compose.ColumnTransformer([("enc1", enc, int_cols), ("scaler1", scaler, real_cols)])
    X_train_new = np.concatenate((X_train_int_new, X_train_real_new),axis=1)
    #print(X_train_new[0])

    # TODO: To the current features, append polynomial features of order 2.
    # If the input values are `[a, b, c, d]`, you should append
    # `[a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2]`. You can generate such polynomial
    # features either manually, or you can employ the provided transformer
    #   sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)
    # which appends such polynomial features of order 2 to the given features.

    poly = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_new)
    #print(X_train_poly[0])

    # TODO: You can wrap all the feature processing steps into one transformer
    # by using `sklearn.pipeline.Pipeline`. Although not strictly needed, it is
    # usually comfortable.

    # TODO: Fit the feature preprocessing steps (the composed pipeline with all of
    # them; or the individual steps, if you prefer) on the training data (using `fit`).
    # Then transform the training data into `train_data` (with a `transform` call;
    # however, you can combine the two methods into a single `fit_transform` call).
    # Finally, transform testing data to `test_data`.
    train_data = ...
    test_data = ...

    return train_data[:5], test_data[:5]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_data, test_data = main(args)
    for dataset in [train_data, test_data]:
        for line in range(min(dataset.shape[0], 5)):
            print(" ".join("{:.4g}".format(dataset[line, column]) for column in range(min(dataset.shape[1], 140))),
                  *["..."] if dataset.shape[1] > 140 else [])