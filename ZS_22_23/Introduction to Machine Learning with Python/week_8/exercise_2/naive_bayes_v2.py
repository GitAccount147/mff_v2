#!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection

# mine:
import sklearn.metrics

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=10, type=float, help="Smoothing parameter of our NB classifier")  # 0.1
parser.add_argument("--naive_bayes_type", default="multinomial", type=str, help="NB type gaussian/multinomial/bernoulli")
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

    class_div = [[] for _ in range(args.classes)]
    for i in range(train_data.shape[0]):
        class_div[train_target[i]].append(train_data[i])
    occurences = [len(item) for item in class_div]
    class_prob = np.log(occurences / np.sum(occurences))

    means = []
    vars = []

    for i in range(args.classes):
        means.append(np.mean(class_div[i], axis=0))
        vars.append(np.var(class_div[i], axis=0) + args.alpha)
    vars = np.sqrt(vars)

    raw_predictions = []
    for i in range(args.classes):
        prob = np.zeros(test_data.shape[0])
        for j in range(len(train_data[i])):
            logpdf = scipy.stats.norm(means[i][j], vars[i][j]).logpdf(test_data.T[j])
            prob += logpdf
        raw_predictions.append(prob)
    raw_predictions = np.array(raw_predictions)
    raw_predictions = raw_predictions.T
    for i in range(test_data.shape[0]):
        raw_predictions[i] += class_prob
    raw_predictions = raw_predictions.T
    prediction = np.argmax(raw_predictions, axis=0)

    acc = sklearn.metrics.accuracy_score(test_target, prediction)

    # multinomial:
    sums = np.zeros((test_data.shape[1], args.classes))
    for i in range(test_data.shape[1]):
        for j in range(args.classes):
            div = np.transpose(np.array(class_div[j]))
            sums[i][j] = np.sum(div[i])
    probs = np.zeros((test_data.shape[1], args.classes))
    for i in range(test_data.shape[1]):
        for j in range(args.classes):
            probs[i][j] = (sums[i][j] + args.alpha)/(np.sum(sums.T[j] + args.alpha))
    weights = np.log(probs)
    bias = class_prob
    raw_predictions = test_data @ weights
    for i in range(len(raw_predictions)):
        raw_predictions[i] += bias
    prediction = np.argmax(raw_predictions, axis=1)
    acc2 = sklearn.metrics.accuracy_score(test_target, prediction)

    # bernoulli:
    divs = []
    for i in range(args.classes):
        div = []
        for j in range(train_data.shape[0]):
            if train_target[j] == i:
                sample = np.zeros(train_data.shape[1])
                for k in range(train_data.shape[1]):
                    if train_data[j][k] >= 8:
                        sample[k] = 1
                div.append(sample)
        divs.append(np.array(div))

    test = np.zeros(test_data.shape)
    for i in range(test_data.shape[0]):
        for j in range(test_data.shape[1]):
            if test_data[i][j] >= 8:
                test[i][j] = 1

    probs = np.zeros((test_data.shape[1], args.classes))
    for i in range(test_data.shape[1]):
        for j in range(args.classes):
            div = np.transpose(divs[j])
            probs[i][j] = (np.sum(div[i]) + args.alpha) / (occurences[j] + 2 * args.alpha)
    weights = np.log(probs / (1 - probs))
    bias = class_prob
    probT = np.transpose(probs)
    probT = np.log(np.ones(probT.shape) - probT)
    for i in range(args.classes):
        bias[i] += np.sum(probT[i])
    raw_pred = test @ weights
    for i in range(len(raw_pred)):
        raw_pred[i] += bias
    pred = np.argmax(raw_pred, axis=1)
    acc3 = sklearn.metrics.accuracy_score(test_target, pred)


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
    if args.naive_bayes_type == 'gaussian':
        test_accuracy = acc
    elif args.naive_bayes_type == 'multinomial':
        test_accuracy = acc2
    else:
        test_accuracy = acc3


    return test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))