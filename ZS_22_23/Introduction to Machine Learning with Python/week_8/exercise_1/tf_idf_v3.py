#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import sys
import urllib.request

import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors

# mine:
import re

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--idf", default=True, action="store_true", help="Use IDF weights")
parser.add_argument("--k", default=15, type=int, help="K nearest neighbors to consider")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=177, type=int, help="Random seed")
parser.add_argument("--tf", default=True, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
parser.add_argument("--train_size", default=2000, type=int, help="Train set size")
# For these and any other arguments you add, ReCodEx will keep your default value.


class NewsGroups:
    def __init__(self,
                 name="20newsgroups.train.pickle",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        with lzma.open(name, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        self.DESCR = dataset.DESCR
        self.data = dataset.data[:data_size]
        self.target = dataset.target[:data_size]
        self.target_names = dataset.target_names


def main(args: argparse.Namespace) -> float:
    # Load the 20newsgroups data.
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)
    # Create train-test split.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=args.seed)
    newsgroups = None  # memory cleanup

    # TODO: Create a feature for every term that is present at least twice
    # in the training data. A term is every maximal sequence of at least 1 word character,
    # where a word character corresponds to a regular expression `\w`.

    # TODO: For each document, compute its features as
    # - term frequency(TF), if `args.tf` is set;
    # - otherwise, use binary indicators (1 if a given term is present, else 0)
    #
    # Then, if `args.idf` is set, multiply the document features by the
    # inverse document frequencies (IDF), where
    # - use the variant which contains `+1` in the denominator;
    # - the IDFs are computed on the train set and then reused without
    #   modification on the test set.

    # TODO: Perform classification of the test set using the k-NN algorithm
    # from sklearn (pass the `algorithm="brute"` option), with `args.k` nearest
    # neighbors. For TF-IDF vectors, the cosine similarity is usually used, where
    #   cosine_similarity(x, y) = x^T y / (||x|| * ||y||).
    #
    # To employ this metric, you have several options:
    # - you could try finding out whether `KNeighborsClassifier` supports it directly;
    # - or you could compute it yourself, but if you do, you have to precompute it
    #   in a vectorized way, so using `metric="precomputed"` is fine, but passing
    #   a callable as the `metric` argument is not (it is too slow);
    # - finally, the nearest neighbors according to cosine_similarity are equivalent to
    #   the neighbors obtained by the usual Euclidean distance on L2-normalized vectors.

    # TODO: Evaluate the performance using a macro-averaged F1 score.

    raw_splitted = []
    for i in range(len(train_data)):
        raw_splitted.append((re.sub(r'\W', r' ', train_data[i])).split())
    train_data = None  # memory cleanup
    splitted = [item for sublist in raw_splitted for item in sublist]

    raw_test_splitted = []
    for i in range(len(test_data)):
        raw_test_splitted.append((re.sub(r'\W', r' ', test_data[i])).split())
    test_data = None  # memory cleanup

    features_raw = {}
    for i in range(len(splitted)):
        if splitted[i] in features_raw:
            features_raw[splitted[i]] += 1
        else:
            features_raw[splitted[i]] = 1

    features_map = {}
    index = 0
    for key in features_raw:
        if features_raw[key] > 1:
            features_map[key] = index
            index += 1
    features = np.concatenate((np.ones((1, len(features_map))),
                               np.zeros((len(raw_splitted) + len(raw_test_splitted), len(features_map)))))
    features_raw = None  # memory cleanup

    for i in range(len(raw_splitted)):
        doc_split = raw_splitted[i]
        length = len(doc_split)
        idf_flag = np.zeros(len(features_map))
        for word in doc_split:
            index = features_map.get(word)
            if index is not None:
                if args.tf:
                    features[i + 1][index] += 1
                else:
                    features[i + 1][index] = 1
                if args.idf:
                    idf_flag[index] = 1
        if args.tf:
            features[i + 1] /= length
        if args.idf:
            features[0] += idf_flag

    if args.idf:
        features[0] = np.log(len(raw_splitted) / features[0])
        for i in range(len(raw_splitted)):
            features[i + 1] *= features[0]

    shift = len(raw_splitted)
    for i in range(len(raw_test_splitted)):
        doc_split = raw_test_splitted[i]
        length = len(doc_split)
        for word in doc_split:
            index = features_map.get(word)
            if index is not None:
                if args.tf:
                    features[i + shift + 1][index] += 1
                else:
                    features[i + shift + 1][index] = 1
        if args.tf and args.idf:
            features[i + shift + 1] = features[i + shift + 1] * features[0] / length
        elif args.tf and not args.idf:
            features[i + shift + 1] /= length
        elif not args.tf and args.idf:
            features[i + shift + 1] *= features[0]

    train_data, test_data = features[1:1+len(raw_splitted)], features[1+len(raw_splitted):]
    test_target = np.array(test_target)
    train_target = np.array(train_target)

    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=args.k, algorithm='brute', metric='cosine')
    knn.fit(train_data, train_target)
    predict = knn.predict(test_data)
    f1_score = sklearn.metrics.f1_score(test_target, predict, average='macro')

    return f1_score


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(args)
    print("F-1 score for TF={}, IDF={}, k={}: {:.1f}%".format(args.tf, args.idf, args.k, 100 * f1_score))