#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

# mine:
import sklearn.neural_network
import sklearn.pipeline
import sklearn.ensemble
import sklearn.preprocessing
import time
from datetime import datetime


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="human_activity_recognition.model", type=str, help="Model path")


class Dataset:
    CLASSES = ["sitting", "sittingdown", "standing", "standingup", "walking"]

    def __init__(self,
                 name="human_activity_recognition.train.csv.xz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and if it contains column "class", split it to `targets`.
        self.data = pd.read_csv(name)
        if "class" in self.data:
            self.target = np.array([Dataset.CLASSES.index(target) for target in self.data["class"]], np.int32)
            self.data = self.data.drop("class", axis=1)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        clf = sklearn.linear_model.SGDClassifier()
        hist_gbc = sklearn.ensemble.HistGradientBoostingClassifier()
        gbc = sklearn.ensemble.GradientBoostingClassifier()
        mlp = sklearn.neural_network.MLPClassifier()
        scaler = sklearn.preprocessing.StandardScaler()
        pipe3 = sklearn.pipeline.Pipeline([("clf", clf)])
        pipe2 = sklearn.pipeline.Pipeline([("mlp", mlp)])
        pipe4 = sklearn.pipeline.Pipeline([("scaler", scaler), ("mlp", mlp)])
        pipe = sklearn.pipeline.Pipeline([("scaler", scaler), ("gbc", gbc)])
        pipe5 = sklearn.pipeline.Pipeline([("gbc", gbc)])
        pipe6 = sklearn.pipeline.Pipeline([("hist_gbc", hist_gbc)])

        n_estimators = [[100, 200, 500], [512], [50], [1024]]
        loss = [['log_loss', 'deviance', 'exponential']]

        max_iter = [[512], [1024], [2048], [64, 128, 256, 512, 1024, 2048]]
        learning_rate = [[0.1], [0.001, 0.01, 0.1, 0.2], [0.2, 0.3], [0.1, 0.15, 0.2, 0.25],
                         np.geomspace(0.0001, 1, num=10), np.geomspace(0.001, 0.5, num=10)]
        learning_rate2 = [np.geomspace(0.1, 0.5, num=10), np.geomspace(0.1, 0.2, num=10)]
        loss_hist = [['log_loss', 'auto']]
        tol =[[10**(-7)], np.geomspace(10**(-9), 10**(-7), num=2), [10**(-9)], [10**(-11)]]
        #class_weight = [[None], ['balanced']]
        parameters = {
            #'gbc__n_estimators': n_estimators[3],
            #'gbc__loss': loss[0],

            'hist_gbc__max_iter': max_iter[2],
            'hist_gbc__learning_rate': learning_rate2[1],
            #'hist_gbc__loss': loss_hist[0],
            'hist_gbc__tol': tol[3],
            #'hist_gbc__class_weight': class_weight[1],
        }


        final_parameters = parameters
        verbose = 10
        model = sklearn.model_selection.GridSearchCV(pipe6, final_parameters, refit=True, cv=None, verbose=verbose)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Started calculation at: ", current_time)

        time_start = time.time()
        model.fit(train.data, train.target)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Done in %0.3fs (= %0.3fh)" % (time.time() - time_start, (time.time() - time_start) / 3600))
        print("Finished at: ", current_time)

        print("Best score: %0.3f" % model.best_score_)
        print("Best parameters set:")
        best_parameters = model.best_estimator_.get_params()
        for param_name in sorted(final_parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))




        # TODO: Train a model on the given dataset and store it in `model`.
        #model = ...

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list or a NumPy array.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)