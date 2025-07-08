#!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter of our NB classifier")
parser.add_argument("--naive_bayes_type", default="gaussian", type=str, help="NB type gaussian/multinomial/bernoulli")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=72, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    print(train_data[0], train_target[0])

    """
    occurences = []
    for i in range(args.classes):
        occurences.append(np.count_nonzero(train_target == i))
    print(occurences)
    """
    class_div = [[] for _ in range(args.classes)]
    for i in range(train_data.shape[0]):
        if train_target[i] <= args.classes:
            class_div[train_target[i]].append(train_data[i])
    occurences = [len(item) for item in class_div]
    #print(class_div[5][:2])
    #print(occurences)

    means = [] #np.zeros(args)
    vars = []

    for i in range(args.classes):
        means.append(np.mean(class_div[i], axis=0))
        vars.append(np.var(class_div[i], axis=0) + args.alpha)

    #print(means[5], vars[5])
    #vars += args.alpha
    #for i in range(640):
    #    a = scipy.stats.norm(0, 1).pdf(test_data)
    #print(a)

    choose = 3
    a = train_data[choose]
    #b = np.sum(scipy.stats.norm(means, vars).logpdf(a))
    print(len(a), len(means[0]), len(vars[0]))

    c = []
    for j in range(args.classes):
        b = []
        for i in range(len(a)):
            b.append(scipy.stats.norm(means[j][i], vars[j][i]).logpdf(a[i]))
        c.append(np.sum(b))
    #b = scipy.stats.norm(means, vars).logpdf(a)
    print(c)
    print(np.argmin(c))
    print("guess vs correct:", np.argmax(c), train_target[choose])

    # pred:
    #prediction = np.argmax(np.sum(scipy.stats.norm.logpdf()))
    predictions = np.zeros((args.classes, ))
    for i in range(args.classes):
        for j in range(len(train_data[0])):
            predictions.append(np.sum(scipy.stats.norm(means[i][j], vars[i][j]).logpdf(test_data[:][j])))
        #print(scipy.stats.norm([0, 0], [1, 1]).logpdf([0, 1, 2]))
    print(predictions[0])

    # TODO: Train a naive Bayes classifier on the train data.
    #
    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    #
    #   During prediction, you can compute the probability density function
    #   of a Gaussian distribution using `scipy.stats.norm`, which offers
    #   `pdf` and `logpdf` methods, among others.
    #
    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.
    #
    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Because Bernoulli NB works with binary data, binarize the features as
    #   [feature_value >= 8], i.e., consider a feature as one iff it is >= 8,
    #   during both estimation and prediction.
    #
    # In all cases, the class prior is the distribution of the train data classes.

    # TODO: Predict the test data classes and compute the test accuracy.
    test_accuracy = ...

    return test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))