#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=9, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.5, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def sigm(num):
    return 1 / (1 + np.exp(-num))


def predict(wts, dat):
    ret = []
    for i in range(len(dat)):
        z = np.dot(wts, dat[i])
        if sigm(z) >= 0.5:
            ret.append(1)
        else:
            ret.append(0)
    return np.array(ret)


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artificial classification dataset
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)

    # TODO: Append a constant feature with value 1 to the end of every input data
    data = np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target \
        = sklearn.model_selection.train_test_split(data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial logistic regression weights
    weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.

        for batch_num in range(train_data.shape[0] // args.batch_size):
            grad_sum = np.zeros(train_data.shape[1])
            for i in range(args.batch_size):
                index = permutation[batch_num * args.batch_size + i]
                grad = (sigm(np.transpose(train_data[index]) @ weights) - train_target[index]) * train_data[index]
                grad_sum += grad
            grad_sum = grad_sum / args.batch_size
            weights = weights - args.learning_rate * grad_sum

        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train set and the test set. The loss is the average MLE loss (i.e., the
        # negative log likelihood, or crossentropy loss, or KL loss) per example.
        train_predict = predict(weights, train_data)
        train_accuracy = sklearn.metrics.accuracy_score(train_target, train_predict)

        train_predict_precise = []
        for i in range(train_data.shape[0]):
            train_predict_precise.append(sigm((train_data @ weights)[i]))

        train_loss = sklearn.metrics.log_loss(train_target, train_predict_precise)

        test_predict = predict(weights, test_data)
        test_accuracy = sklearn.metrics.accuracy_score(test_target, test_predict)

        test_predict_precise = []
        for i in range(test_data.shape[0]):
            test_predict_precise.append(sigm((test_data @ weights)[i]))
        test_loss = sklearn.metrics.log_loss(test_target, test_predict_precise)

        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

        if args.plot:
            import matplotlib.pyplot as plt
            if args.plot is not True:
                plt.gcf().get_axes() or plt.figure(figsize=(6.4*3, 4.8*(args.epochs+2)//3))
                plt.subplot(3, (args.epochs+2)//3, 1 + epoch)
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[1 / (1 + np.exp(-([x, y, 1] @ weights))) for x in xs] for y in ys]
            plt.contourf(xs, ys, predictions, levels=21, cmap="RdBu", alpha=0.7)
            plt.contour(xs, ys, predictions, levels=[0.25, 0.5, 0.75], colors="k")
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, label="train", marker="P", cmap="RdBu")
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, label="test", cmap="RdBu")
            plt.legend(loc="upper right")
            plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights", *("{:.2f}".format(weight) for weight in weights))