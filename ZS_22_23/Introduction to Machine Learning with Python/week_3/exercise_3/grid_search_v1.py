#!/usr/bin/env python3
import argparse
import warnings

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load digit dataset
    dataset = sklearn.datasets.load_digits()
    dataset.target = dataset.target % 2

    data, target = dataset.data, dataset.target

    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target \
        = sklearn.model_selection.train_test_split(data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Create a pipeline, which
    # 1. passes the inputs through `sklearn.preprocessing.MinMaxScaler()`,
    # 2. passes the result through `sklearn.preprocessing.PolynomialFeatures()`,
    # 3. passes the result through `sklearn.linear_model.LogisticRegression(random_state=args.seed)`.
    #

    scaler = sklearn.preprocessing.MinMaxScaler()
    poly = sklearn.preprocessing.PolynomialFeatures()
    log_reg = sklearn.linear_model.LogisticRegression(random_state=args.seed)
    pipe = sklearn.pipeline.Pipeline([('scale', scaler), ('poly_feat', poly), ('log_reg', log_reg)])

    parameters = [{'poly_feat__degree': [1, 2], 'log_reg__C': [0.01, 1, 100], 'log_reg__solver': ['lbfgs', 'sag']}]

    warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

    skf = sklearn.model_selection.StratifiedKFold(n_splits=5)
    model = sklearn.model_selection.GridSearchCV(pipe, param_grid=parameters, cv=skf, refit=True)
    model.fit(train_data, train_target)
    for rank, accuracy, params in zip(model.cv_results_["rank_test_score"],
                                      model.cv_results_["mean_test_score"], model.cv_results_["params"]):
        print("Rank: {:2d} Cross-val: {:.1f}%".format(rank, 100 * accuracy),
              *("{}: {:<5}".format(key, value) for key, value in params.items()))

    test_accuracy = sklearn.metrics.accuracy_score(test_target, model.predict(test_data))

    # Then, using `sklearn.model_selection.StratifiedKFold(5)`, evaluate crossvalidated
    # train performance of all combinations of the following parameters:
    # - polynomial degree: 1, 2
    # - LogisticRegression regularization C: 0.01, 1, 100
    # - LogisticRegression solver: lbfgs, sag
    #
    # For the best combination of parameters, compute the test set accuracy.
    #
    # The easiest way is to use `sklearn.model_selection.GridSearchCV`.

    # If `model` is a fitted `GridSearchCV`, you can use the following code
    # to show the results of all the hyperparameter values evaluated:
    #   for rank, accuracy, params in zip(model.cv_results_["rank_test_score"],
    #                                     model.cv_results_["mean_test_score"],
    #                                     model.cv_results_["params"]):
    #       print("Rank: {:2d} Cross-val: {:.1f}%".format(rank, 100 * accuracy),
    #             *("{}: {:<5}".format(key, value) for key, value in params.items()))

    # Note that with some hyperparameter values above, the training does not
    # converge in the default limit of 100 epochs and shows `ConvergenceWarning`s.
    # You can verify that increasing the number of epochs influences the results
    # only marginally, so there is no reason to do it. To get rid of the warnings,
    # you can add `-W ignore::UserWarnings` just after `python` on the command line,
    # or you can use the following code (and the corresponding imports):
    #   warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

    return test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)
    print("Test accuracy: {:.2f}".format(100 * test_accuracy))