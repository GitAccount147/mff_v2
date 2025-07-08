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

# mine:
import sklearn.model_selection
import time
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.pipeline
import sklearn.neural_network
from datetime import datetime

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="isnt_it_ironic.model", type=str, help="Model path")


class Dataset:
    def __init__(self,
                 name="isnt_it_ironic.train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            for line in dataset_file:
                label, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.target.append(int(label))
        self.target = np.array(self.target, np.int32)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()


        tf_idf_vect = sklearn.feature_extraction.text.TfidfVectorizer()
        clf = sklearn.linear_model.SGDClassifier()
        mlp = sklearn.neural_network.MLPClassifier()
        pipe = sklearn.pipeline.Pipeline([("tfidfvect", tf_idf_vect), ("clf", clf)])
        pipe2 = sklearn.pipeline.Pipeline([("tfidfvect", tf_idf_vect), ("mlp", mlp)])
        parameters = {
            # "tfidfvect__max_df": (0.5, 0.75, 1.0),
            # 'tfidfvect__max_features': (None, 5000, 10000, 50000),
            "tfidfvect__ngram_range": ((1, 1), (1, 2), (1, 3)),  # unigrams or bigrams
            # 'tfidfvect__use_idf': (True, False),
            'tfidfvect__norm': ('l1', 'l2'),
            "clf__max_iter": (1000, 2000,),
            "clf__alpha": (0.00001, 0.000001),
            "clf__penalty": ("l2", "elasticnet"),
            # 'clf__max_iter': (10, 50, 80),
            "clf__loss": ("hinge", "log_loss")
        }
        lowercase = [(True,), (True, False)]
        analyzer = [('word',), ('word', 'char', 'char_wb'), ('char_wb',)]
        stop_words = [(None, ), (None, 'english')]
        ngram_range = [((1, 4),), ((1, 1), (1, 2)), ((1, 1), (1, 2), (1, 3)), ((1, 1), (1, 2), (1, 3), (1, 4))]
        ngram_range2 = [((1, 4), (1, 5), (1, 6))]
        max_df = [(1.0, ), (0.5, 0.625, 0.75, 0.875, 1.0)]
        min_df = [(1,), (1, 2, 3)]
        binary = [(False,), (False, True)]
        norm = [('l2',), ('l2', 'l1'), ('l1',)]
        use_idf = [(True,), (True, False)]
        smooth_idf = [(True,), (True, False)]
        sublinear_tf = [(False,), (False, True)]
        loss = [('hinge',), ('hinge', 'log_loss', 'modified_huber')]
        penalty = [('l2',), ('l2', 'l1', 'elasticnet')]
        alpha = [(0.0001,), (0.001, 0.001, 0.00001, 0.000001)]
        max_iter = [(1000,), (1000, 2000)]
        tol = [(0.001,), (0.001, 0.0001)]
        parameters_3 = {
            #'tfidfvect__lowercase': lowercase[1],
            'tfidfvect__analyzer': analyzer[2],
            #'tfidfvect__stop_words': stop_words[0],
            'tfidfvect__ngram_range': ngram_range[0],
            #'tfidfvect__max_df': max_df[0],
            #'tfidfvect__min_df': min_df[0],
            'tfidfvect__norm': norm[2],


            #'tfidfvect__binary': binary[1],
            #'tfidfvect__use_idf': use_idf[1],
            #'tfidfvect__smooth_idf': smooth_idf[1],
            #'tfidfvect__sublinear_tf': sublinear_tf[1],

            #'clf__loss': loss[1],
            #'clf__penalty': penalty[1],
            #'clf__alpha': alpha[1],
            'clf__alpha': (0.00001, 0.000001),
            'clf__max_iter': max_iter[1],

            #'clf__tol': tol[1]
        }
        parameters2 = {
            # "tfidfvect__max_df": (0.5, 0.75, 1.0),
            # 'tfidfvect__max_features': (None, 5000, 10000, 50000),
            # "tfidfvect__ngram_range": ((1, 1), (1, 2), (1, 3)),  # unigrams or bigrams
            # 'tfidfvect__use_idf': (True, False),
            'tfidfvect__norm': ('l2', 'l1'),
            "mlp__max_iter": (800,),
            "mlp__hidden_layer_sizes": ((800,), (400,))
        }
        verbose = 10
        model = sklearn.model_selection.GridSearchCV(pipe, parameters_3, refit=True, cv=None, verbose=verbose)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Started calculation at: ", current_time)

        time_start = time.time()
        model.fit(train.data, train.target)
        print("done in %0.3fs" % (time.time() - time_start))

        print("Best score: %0.3f" % model.best_score_)
        print("Best parameters set:")
        best_parameters = model.best_estimator_.get_params()
        for param_name in sorted(parameters_3.keys()):
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