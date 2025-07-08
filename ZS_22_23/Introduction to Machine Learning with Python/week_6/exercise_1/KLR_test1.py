#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.metrics
import sklearn.metrics.pairwise

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("--data_size", default=50, type=int, help="Data size")
parser.add_argument("--epochs", default=200, type=int, help="Number of SGD training epochs")
parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=3, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization weight")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.


def ker_poly_hand(data1, data2, deg, gamma):
    res = np.zeros((data1.shape[0], data2.shape[0]))
    gram = data1 @ np.transpose(data2)
    for i in range(data1.shape[0]):
        for j in range(data2.shape[0]):
            res[i][j] = np.power(gamma * gram[i][j] + 1, deg)
    return res


def ker_rbf_hand(data1, data2, gamma):
    res = np.zeros((data1.shape[0], data2.shape[0]))
    print(res.shape, data1.shape, data2.shape, data1.shape[0], data2.shape[0])
    for i in range(data1.shape[0]):
        for j in range(data2.shape[0]):
            res[i][j] = np.exp(- gamma * np.power(np.linalg.norm(data1[i][0] - data2[j][0]), 2))
    return res


def main(args: argparse.Namespace) -> tuple[np.ndarray, float, list[float], list[float]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial regression dataset.
    train_data = np.linspace(-1, 1, args.data_size)
    train_target = np.sin(5 * train_data) + generator.normal(scale=0.25, size=args.data_size) + 1

    train_data = train_data.reshape(-1, 1)

    test_data = np.linspace(-1.2, 1.2, 2 * args.data_size)
    test_target = np.sin(5 * test_data) + 1

    test_data = test_data.reshape(-1, 1)

    # Initialize the parameters: the betas and the bias.
    betas = np.zeros(args.data_size)
    bias = 0

    # compute the kernel:
    ker_train = ker_poly_hand(train_data, train_data, args.kernel_degree, args.kernel_gamma)
    ker_test = ker_poly_hand(train_data, test_data, args.kernel_degree, args.kernel_gamma)
    if args.kernel == "rbf":
        ker_train = ker_rbf_hand(train_data, train_data, args.kernel_gamma)
        ker_test = ker_rbf_hand(train_data, test_data, args.kernel_gamma)

    train_rmses, test_rmses = [], []
    test_predictions = []
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])
        for i in range(train_data.shape[0] // args.batch_size):
            grad_betas = np.zeros(args.data_size)
            grad_bias = 0
            for j in range(args.batch_size):
                index = permutation[i * args.batch_size + j]

                mini_sum = 0
                for k in range(args.data_size):
                    mini_sum += betas[k] * ker_train[index][k]

                grad_betas[index] += mini_sum - train_target[index]
                grad_bias += grad_betas[index] + bias
            betas -= (args.learning_rate / args.batch_size) * grad_betas
            betas -= args.learning_rate * args.l2 * betas
            bias -= (args.learning_rate / args.batch_size) * grad_bias
            bias -= args.learning_rate * args.l2 * bias
        train_prediction = ker_train.T @ betas + bias
        test_prediction = ker_test.T @ betas + bias
        test_predictions = test_prediction

        train_rmse = sklearn.metrics.mean_squared_error(train_target, train_prediction, squared=False)
        test_rmse = sklearn.metrics.mean_squared_error(test_target, test_prediction, squared=False)

        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)

        if (epoch + 1) % 10 == 0:
            print("After epoch {}: train RMSE {:.2f}, test RMSE {:.2f}".format(
                epoch + 1, train_rmses[-1], test_rmses[-1]))

# first example:
# -17.58 -11.77 0.17 10.94 6.38 7.94 18.47 14.35 7.00 10.15 2.17 0.51 1.74 -11.90 -13.97 ...
# Learned bias 0.48665425877260254

# first rbf example:
# Learned betas 0.65 0.59 1.17 1.72 0.86 0.82 1.61 1.04 0.21 0.47 -0.31 -0.56 -0.46 -1.77 -1.88 ...
# Learned bias 0.6512539914766637

    if args.plot:
        import matplotlib.pyplot as plt
        # If you want the plotting to work (not required for ReCodEx), compute the `test_predictions`.
        test_predictions = test_predictions

        plt.plot(train_data, train_target, "bo", label="Train target")
        plt.plot(test_data, test_target, "ro", label="Test target")
        plt.plot(test_data, test_predictions, "g-", label="Predictions")
        plt.legend()
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return betas, bias, train_rmses, test_rmses


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    betas, bias, train_rmses, test_rmses = main(args)
    print("Learned betas", *("{:.2f}".format(beta) for beta in betas[:15]), "...")
    print("Learned bias", bias)