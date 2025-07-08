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
parser.add_argument("--k", default=1, type=int, help="K nearest neighbors to consider")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=177, type=int, help="Random seed")
parser.add_argument("--tf", default=True, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")
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

    # TODO: Create a feature for every term that is present at least twice
    # in the training data. A term is every maximal sequence of at least 1 word character,
    # where a word character corresponds to a regular expression `\w`.

    a = re.sub(r'\W', r' ', "dvbj jsbnij)( jibv 7b*/d ")
    print(a, a.split())

    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    splitted = []
    for i in range(len(train_data)):
        word = ""
        for j in range(len(train_data[i])):
            if train_data[i][j] in chars:
                word += train_data[i][j]
            else:
                if word != "":
                    #splitted.append(word.lower())
                    splitted.append(word)
                word = ""
        if word != "":
            splitted.append(word)
    #print(splitted[:])
    #print((train_data[0].split("\w"))[0])

    features_raw = {}
    for i in range(len(splitted)):
        if splitted[i] in features_raw:
            features_raw[splitted[i]] += 1
        else:
            features_raw[splitted[i]] = 1

    features = {}
    for key in features_raw:
        if features_raw[key] > 1:  # ? >= 1
            #features[key] = features_raw[key]
            features[key] = 0

    # memory_util:
    features_raw = {}

    #print(features)


    # TODO: For each document, compute its features as
    # - term frequency(TF), if `args.tf` is set;
    # - otherwise, use binary indicators (1 if a given term is present, else 0)
    #

    all_docs = []
    idf = {}
    for key in features:
        idf[key] = 1

    for i in range(len(train_data)):
        doc_features = {}
        for key in features:
            doc_features[key] = 0
        word = ""
        length = 0
        for j in range(len(train_data[i])):
            if train_data[i][j] in chars:
                word += train_data[i][j]
            else:
                if word != "":
                    #length += 1
                    #if word.lower() in doc_features:
                    if word in doc_features:
                        length += 1
                        if args.tf:
                            #doc_features[word.lower()] += 1
                            doc_features[word] += 1
                        else:
                            #doc_features[word] -= doc_features[word] - 1
                            #doc_features[word.lower()] = 1
                            doc_features[word] = 1
                    word = ""
        if word != "":
            if word in doc_features:
                length += 1
                if args.tf:
                    doc_features[word] += 1
                else:
                    doc_features[word] = 1
        if args.tf:
            for key in doc_features:
                doc_features[key] /= length
        """
        # wrong?:
        else:
            for key in doc_features:
                if doc_features[key] == 1:
                    doc_features[key] = 0
        """
        if args.idf:
            for key in doc_features:
                if doc_features[key] > 0:
                    idf[key] += 1
        all_docs.append(doc_features)
    #print(all_docs[1])
    #print(idf)




    # Then, if `args.idf` is set, multiply the document features by the
    # inverse document frequencies (IDF), where
    # - use the variant which contains `+1` in the denominator;
    # - the IDFs are computed on the train set and then reused without
    #   modification on the test set.

    for key in idf:
        idf[key] = np.log(len(train_data) / (idf[key]))
    #print(idf)

    # memory_util:
    train_data = []

    if args.idf:
        for doc in all_docs:
            for key in doc:
                doc[key] *= idf[key]

    #print(all_docs[0])


    data = []
    for i in range(len(all_docs)):
        data.append(list(all_docs[i].values()))

    # memory_util:
    all_docs = []

    data = np.array(data)
    new_train_target = np.array(train_target)

    test_all_docs = []
    for i in range(len(test_data)):
        doc_features = {}
        for key in features:
            doc_features[key] = 0
        word = ""
        length = 0
        for j in range(len(test_data[i])):
            if test_data[i][j] in chars:
                word += test_data[i][j]
            else:
                if word != "":
                    #length += 1
                    #if word.lower() in doc_features:
                    if word in doc_features:
                        length += 1
                        if args.tf:
                            #doc_features[word.lower()] += 1
                            doc_features[word] += 1
                        else:
                            #doc_features[word] -= doc_features[word] - 1
                            #doc_features[word.lower()] = 1
                            doc_features[word] = 1
                    word = ""
        if word != "":
            if word in doc_features:
                length += 1
                if args.tf:
                    doc_features[word] += 1
                else:
                    doc_features[word] = 1
        if args.tf:
            for key in doc_features:
                doc_features[key] /= length
        """
        else:
            for key in doc_features:
                if doc_features[key] == 1:
                    doc_features[key] = 0
        """
        test_all_docs.append(doc_features)

    # memory_util:
    test_data = []

    if args.idf:
        for doc in test_all_docs:
            for key in doc:
                doc[key] *= idf[key]

    test_data = []
    for i in range(len(test_all_docs)):
        test_data.append(list(test_all_docs[i].values()))

    # memory_util:
    test_all_docs = []

    test_data = np.array(test_data)
    new_test_target = np.array(test_target)

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

    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=args.k, algorithm='brute', metric='cosine')
    knn.fit(data, new_train_target)
    predict = knn.predict(test_data)

    # TODO: Evaluate the performance using a macro-averaged F1 score.
    f1_score = sklearn.metrics.f1_score(new_test_target, predict, average='macro')

    return f1_score


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(args)
    print("F-1 score for TF={}, IDF={}, k={}: {:.1f}%".format(args.tf, args.idf, args.k, 100 * f1_score))