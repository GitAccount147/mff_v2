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

# mine:
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.pipeline
import sklearn.linear_model
import sklearn.neural_network
import warnings
import sklearn.neighbors
import sklearn.ensemble
# end mine

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")


class Dataset:
    """MNIST Dataset.
    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in the range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(float)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # mega-param:
        folds = 5

        warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

        # hyper-param:
        degree_range = [[1, 2, 3], [1, 2], [2], [1]]
        alpha_range = [[0.01], [0.001, 0.01, 0.1], 10.0 ** - np.arange(-2, 7), [1, 10, 100], [0.1]]
        layer_sizes = [[(100,)], [(200,)], [(50,)], [(50,), (100,), (200,)], [(10, 10), (50, 50), (100, 100)],
                       [(400, ), (200, 200), (400, 200, 100), (400, 200, 100, 50)], [(400, 400), (800,), (800, 800)]]
        layer_sizes2 = [[(512,), (128, 256)], [(512,)], [(128, 512)], [(256, 512)], [(1024,)], [(2048,), (4096,)]]
        layer_sizes3 = [[(128, 128, 128, 128, 128), (128, 128, 128, 128, 128, 128, 128, 128, 128, 128),
                         (256, 256, 256, 256, 256), (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)]]
        layer_sizes4 = [[(128,), (256,), (512,)]]
        activation = [['relu'], ['logistic', 'tanh', 'relu']]
        solver = [['adam'], ['adam', 'sgd', 'lbfgs']]
        learning_rate = [['constant'], ['constant', 'invscaling', 'adaptive']]
        learning_rate_init = [[0.001], [0.01, 0.001, 0.0001], [0.0001]]
        max_iter = [[200], [200, 400], [800]]

        # for testing:
        #test_size = 0.9
        #train_data, test_data, train_target, test_target \
        #    = sklearn.model_selection.train_test_split(train.data, train.target, test_size=test_size, random_state=args.seed)

        scaler = sklearn.preprocessing.StandardScaler()
        #scaler =sklearn.preprocessing.RobustScaler()
        #poly = sklearn.preprocessing.PolynomialFeatures()  #('poly', poly)
        #spline = sklearn.preprocessing.SplineTransformer()
        #spline = sklearn.preprocessing.SplineTransformer(degree=2, n_knots=5, knots='quantile')
        #power = sklearn.preprocessing.PowerTransformer()
        mlp = sklearn.neural_network.MLPClassifier(random_state=args.seed)
        #pipe = sklearn.pipeline.Pipeline([('spline', spline), ('mlp', mlp)])
        pipe = sklearn.pipeline.Pipeline([('scaler', scaler), ('mlp', mlp)])
        #pipe = sklearn.pipeline.Pipeline([('power', power), ('mlp', mlp)])
        #pipe = sklearn.pipeline.Pipeline([('mlp', mlp)])

        parameters = [{'mlp__alpha': alpha_range[0], 'mlp__hidden_layer_sizes': layer_sizes2[5],
                       'mlp__activation': activation[0], 'mlp__solver': solver[0],
                       'mlp__learning_rate': learning_rate[0], 'mlp__learning_rate_init': learning_rate_init[0],
                       'mlp__max_iter': max_iter[2]}]

        warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

        skf = sklearn.model_selection.StratifiedKFold(n_splits=folds)
        model = sklearn.model_selection.GridSearchCV(pipe, param_grid=parameters, cv=skf, refit=True, verbose=4)
        #model = sklearn.model_selection.GridSearchCV(pipe, param_grid=parameters, cv=skf, refit=True)
        model.fit(train.data, train.target)
        print("best score:", model.best_score_ * 100, "%, while using:", model.best_params_)

        """
        neig = sklearn.neighbors.KNeighborsClassifier()
        gbdt = sklearn.ensemble.GradientBoostingClassifier()
        pipe2 = sklearn.pipeline.Pipeline([('scaler', scaler), ('neig', neig)])
        pipe3 = sklearn.pipeline.Pipeline([('scaler', scaler), ('gbdt', gbdt)])
        param2 = [{'neig__n_neighbors': [5, 10, 2]}]
        param3 = [{'gbdt__learning_rate': [0.1, 0.01]}]
        model2 = sklearn.model_selection.GridSearchCV(pipe3, param_grid=param3, cv=skf, refit=True, verbose=4)
        model2.fit(train.data, train.target)
        print("GBDT: best score:", model2.best_score_ * 100, "%, while using:", model2.best_params_)
        """


        """
        for rank, accuracy, params in zip(model.cv_results_["rank_test_score"],
                                          model.cv_results_["mean_test_score"], model.cv_results_["params"]):
            print("Rank: {:2d} Cross-val: {:.1f}%".format(rank, 100 * accuracy),
                  *("{}: {:<5}".format(key, value) for key, value in params.items()))
        """




        # TODO: Train a model on the given dataset and store it in `model`.
        #model = ...

        # If you trained one or more MLPs, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained `MLPClassifier` is in the `mlp` variable.

        """
        print(mlp.coefs_)
        mlp._optimizer = None
        for i in range(len(mlp.coefs_)):
            mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        for i in range(len(mlp.intercepts_)):
            mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)
        """

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