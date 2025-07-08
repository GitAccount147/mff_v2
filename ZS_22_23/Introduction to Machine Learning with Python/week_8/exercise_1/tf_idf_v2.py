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
import tracemalloc

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--idf", default=True, action="store_true", help="Use IDF weights")
parser.add_argument("--k", default=1, type=int, help="K nearest neighbors to consider")
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
    tracemalloc.start()
    # Load the 20newsgroups data.
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)
    #print(tracemalloc.get_traced_memory()[0] / 1000000)
    #tracemalloc.stop()

    #tracemalloc.start()
    # Create train-test split.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=args.seed)
    # mem-ut:
    #print(tracemalloc.get_traced_memory()[0] / 1000000)
    #tracemalloc.stop()
    newsgroups = []

    a = re.sub(r'\W', r' ', "dvbj jsbnij)( jibv 7b*/d ")
    print(a, a.split())

    raw_splitted = []
    for i in range(len(train_data)):
        raw_splitted.append((re.sub(r'\W', r' ', train_data[i])).split())
    # memory_util:
    train_data = []
    splitted = [item for sublist in raw_splitted for item in sublist]
    #print(tracemalloc.get_traced_memory()[0] / 1000000)

    raw_test_splitted = []
    for i in range(len(test_data)):
        raw_test_splitted.append((re.sub(r'\W', r' ', test_data[i])).split())
    # memory-util:
    test_data = []
    #print(tracemalloc.get_traced_memory()[0] / 1000000)

    features_raw = {}
    for i in range(len(splitted)):
        if splitted[i] in features_raw:
            features_raw[splitted[i]] += 1
        else:
            features_raw[splitted[i]] = 1

    features = {}
    for key in features_raw:
        if features_raw[key] > 1:
            features[key] = 0

    features_raw = {}  # memory_util:

    all_docs = []
    idf = dict.fromkeys(features, 1)
    print(tracemalloc.get_traced_memory()[1] / 1000000)
    for i in range(len(raw_splitted)):
        doc_features = dict.fromkeys(features, 0)
        #"""
        doc_split = raw_splitted[i]
        #"""
        length = len(doc_split)
        for word in doc_split:
            if word in doc_features:
                if args.tf:
                    doc_features[word] += 1
                else:
                    doc_features[word] = 1
        if args.tf:
            for key in doc_features:
                doc_features[key] /= length
        if args.idf:
            for key in doc_features:
                if doc_features[key] > 0:
                    idf[key] += 1
        #"""
        all_docs.append(doc_features)
    print(tracemalloc.get_traced_memory()[1] / 1000000)

    #print(tracemalloc.get_traced_memory()[0] / 1000000)

    for key in idf:
        idf[key] = np.log(len(raw_splitted) / (idf[key]))

    if args.idf:
        for doc in all_docs:
            for key in doc:
                doc[key] *= idf[key]

    data = []
    for i in range(len(all_docs)):
        data.append(list(all_docs[i].values()))

    # memory_util:
    all_docs = []

    #print(tracemalloc.get_traced_memory()[0] / 1000000)

    data = np.array(data)
    new_train_target = np.array(train_target)

    test_data = []
    for i in range(len(raw_test_splitted)):
        doc_features = dict.fromkeys(features, 0)
        doc_split = raw_test_splitted[i]
        length = len(doc_split)
        for word in doc_split:
            if word in doc_features:
                if args.tf:
                    doc_features[word] += 1
                else:
                    doc_features[word] = 1
        if args.tf and not args.idf:
            for key in doc_features:
                doc_features[key] /= length
        elif args.tf and args.idf:
            for key in doc_features:
                doc_features[key] *= idf[key] / length
        elif not args.tf and args.idf:
            for key in doc_features:
                doc_features[key] *= idf[key]
        test_data.append(list(doc_features.values()))

    print(tracemalloc.get_traced_memory()[1] / 1000000)

    test_data = np.array(test_data)
    new_test_target = np.array(test_target)
    tracemalloc.stop()


    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=args.k, algorithm='brute', metric='cosine')
    knn.fit(data, new_train_target)
    predict = knn.predict(test_data)
    f1_score = sklearn.metrics.f1_score(new_test_target, predict, average='macro')

    return f1_score


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(args)
    print("F-1 score for TF={}, IDF={}, k={}: {:.1f}%".format(args.tf, args.idf, args.k, 100 * f1_score))