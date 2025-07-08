#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import scipy.special

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=5, type=int, help="Number of classes to use")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def calculate_metrics(truth, predict):
    avg_f1, total_tp, total_fp_fn = 0, 0, 0
    for c in range(truth.shape[1]):
        current_tp, current_fp_fn = 0, 0
        for i in range(truth.shape[0]):
            if truth[i][c] == 1 and predict[i][c] == 1:
                current_tp += 1
                total_tp += 1
            elif truth[i][c] != predict[i][c]:
                current_fp_fn += 1
                total_fp_fn += 1
        avg_f1 += 2 * current_tp / (2 * current_tp + current_fp_fn)
    f1_macro = avg_f1 / truth.shape[1]
    f1_micro = 2 * total_tp / (2 * total_tp + total_fp_fn)
    return f1_micro, f1_macro


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial classification dataset.
    data, target_list = sklearn.datasets.make_multilabel_classification(
        n_samples=args.data_size, n_classes=args.classes, allow_unlabeled=False,
        return_indicator=False, random_state=args.seed)

    # TODO: The `target` is a list of classes for every input example. Convert
    # it to a dense representation (n-hot encoding) -- for each input example,
    # the target should be vector of `args.classes` binary indicators.
    mlb = sklearn.preprocessing.MultiLabelBinarizer()
    target = mlb.fit_transform(target_list)

    # Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        for batch_num in range(train_data.shape[0] // args.batch_size):
            grad_sum = np.zeros((train_data.shape[1], args.classes))
            for i in range(args.batch_size):
                index = permutation[batch_num * args.batch_size + i]
                soft = scipy.special.expit(train_data[index] @ weights)
                prac = soft - train_target[index]
                grad = (np.outer(prac, train_data[index])).T
                grad_sum += grad
            grad_sum = grad_sum / args.batch_size
            weights = weights - args.learning_rate * grad_sum

        # TODO: After the SGD epoch, compute the micro-averaged and the
        # macro-averaged F1-score for both the train test and the test set.
        # Compute these scores manually, without using `sklearn.metrics`.

        train_precise_predict = np.zeros((train_data.shape[0], args.classes))
        rough_train = scipy.special.expit(train_data @ weights)
        for i in range(train_data.shape[0]):
            for j in range(args.classes):
                if rough_train[i][j] >= 0.5:
                    train_precise_predict[i][j] = 1
        train_f1_micro, train_f1_macro = calculate_metrics(train_target, train_precise_predict)

        test_precise_predict = np.zeros((test_data.shape[0], args.classes))
        rough_test = scipy.special.expit(test_data @ weights)
        for i in range(test_data.shape[0]):
            for j in range(args.classes):
                if rough_test[i][j] >= 0.5:
                    test_precise_predict[i][j] = 1
        test_f1_micro, test_f1_macro = calculate_metrics(test_target, test_precise_predict)

        print("After epoch {}: train F1 micro {:.2f}% macro {:.2f}%, test F1 micro {:.2f}% macro {:.1f}%".format(
            epoch + 1, 100 * train_f1_micro, 100 * train_f1_macro, 100 * test_f1_micro, 100 * test_f1_macro))

    return weights, [(100 * train_f1_micro, 100 * train_f1_macro), (100 * test_f1_micro, 100 * test_f1_macro)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")