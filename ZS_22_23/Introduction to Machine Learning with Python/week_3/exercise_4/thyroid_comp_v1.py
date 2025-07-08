#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
import sklearn.preprocessing

# mine:
import sklearn.model_selection
import sklearn.pipeline
import sklearn.linear_model
import warnings
# end mine

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")


class Dataset:
    """Thyroid Dataset.
    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features
    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        data, target = train.data, train.target

        # mega-param:
        test_size = 0.01
        folds = 5

        train_data, test_data, train_target, test_target \
            = sklearn.model_selection.train_test_split(data, target, test_size=test_size, random_state=args.seed)

        warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


        C_range = np.geomspace(10, 1000, num=100)
        C_range_big = np.geomspace(0.001, 1000, num=1000)
        C_range_basic = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        C_range_mid = np.geomspace(1, 100, num=20)
        C_range_mid2 = np.geomspace(10, 50, num=20)
        degree_range_basic = [1, 2]
        degree_range = [1, 2, 3]
        degree_range_one = [2]
        solver_types = ['lbfgs', 'sag']
        solver_types_one = ['lbfgs']

        scaler = sklearn.preprocessing.StandardScaler()
        poly = sklearn.preprocessing.PolynomialFeatures()
        log_reg_sgd = sklearn.linear_model.SGDClassifier(loss='log_loss', random_state=args.seed)
        log_reg2 = sklearn.linear_model.LogisticRegression(max_iter=1000, random_state=args.seed)
        pipe = sklearn.pipeline.Pipeline([('scaler', scaler), ('poly', poly), ('log_reg', log_reg2)])

        parameters = [{'poly__degree': degree_range_one, 'log_reg__C': C_range, 'log_reg__solver': solver_types_one}]
        parameters2 = [{'poly__degree': [1, 2, 3], 'log_reg__alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]}]

        warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

        skf = sklearn.model_selection.StratifiedKFold(n_splits=folds)
        model = sklearn.model_selection.GridSearchCV(pipe, param_grid=parameters, cv=skf, refit=True)
        model.fit(train_data, train_target)
        for rank, accuracy, params in zip(model.cv_results_["rank_test_score"],
                                          model.cv_results_["mean_test_score"], model.cv_results_["params"]):
            print("Rank: {:2d} Cross-val: {:.1f}%".format(rank, 100 * accuracy),
                  *("{}: {:<5}".format(key, value) for key, value in params.items()))

        pred = model.predict(test_data)
        #print(pred[-100:])
        test_accuracy = sklearn.metrics.accuracy_score(test_target, pred)
        test_loss = sklearn.metrics.log_loss(test_target, pred)
        print(model.best_score_)
        print(model.best_params_)
        print("test accur, test loss:", test_accuracy, test_loss)

        # TODO: Train a model on the given dataset and store it in `model`.
        #model = ...

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)