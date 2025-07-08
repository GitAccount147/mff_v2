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
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=3, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization weight")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate")
parser.add_argument("--plot", default=True, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.


def ker_poly_hand(data1, data2, deg, gamma):
    #print(data1.shape, data2.shape)
    res = np.zeros((data1.shape[0], data2.shape[0]))
    gram = data1 @ np.transpose(data2)
    #print(gram.shape)
    #print(gram)
    for i in range(data1.shape[0]):
        for j in range(data2.shape[0]):
            res[i][j] = np.power(gamma * gram[i][j] + 1, deg)
            #res[i][j] = np.power(gamma * gram[i][j] + 1, deg) + 1
            #res[i][j] = np.power(gamma * gram[i][j] + 1, deg) - 1
    return res


def ker_rbf_hand(data1, data2, gamma):
    res = np.zeros((data1.shape[0], data2.shape[0]))
    print(res.shape, data1.shape, data2.shape, data1.shape[0], data2.shape[0])
    for i in range(data1.shape[0]):
        for j in range(data2.shape[0]):
            res[i][j] = np.exp(- gamma * np.power(np.linalg.norm(data1[i][0] - data2[j][0]), 2)) + 1
    return res


def main(args: argparse.Namespace) -> tuple[np.ndarray, float, list[float], list[float]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial regression dataset.
    train_data = np.linspace(-1, 1, args.data_size)
    train_target = np.sin(5 * train_data) + generator.normal(scale=0.25, size=args.data_size) + 1

    train_data = train_data.reshape(-1, 1)
    #print(train_data.shape)
    print(train_target.shape)
    #print(train_data.T)

    test_data = np.linspace(-1.2, 1.2, 2 * args.data_size)
    test_target = np.sin(5 * test_data) + 1

    test_data = test_data.reshape(-1, 1)
    #print(test_data.shape)
    #print(test_target.shape)
    #print(test_data.T)

    # Initialize the parameters: the betas and the bias.
    betas = np.zeros(args.data_size)
    bias = 0


    # old compute the kernel for train and test:
    """
    #ker_train = sklearn.metrics.pairwise.polynomial_kernel(X=train_data, Y=train_data, degree=args.kernel_degree,
    #                                                       gamma=args.kernel_gamma, coef0=1)
    #ker_test = sklearn.metrics.pairwise.polynomial_kernel(test_data, train_data,
    #                                                      degree=args.kernel_degree, gamma=args.kernel_gamma)
    #if args.kernel == "rbf":
    #    ker_train = sklearn.metrics.pairwise.rbf_kernel(train_data, gamma=args.kernel_gamma)
    #    ker_test = sklearn.metrics.pairwise.rbf_kernel(test_data, train_data, gamma=args.kernel_gamma)
    #print(ker_train.shape)
    #print(ker_train[0], ker_train[-1])
    #print(ker_train.T[0], ker_train.T[-1])
    #print(ker_train - ker_train.T)
    #ker_train[0][-1] = 1
    #ker_train[-1][0] = 1
    """

    # compute the kernel:
    ker_train = ker_poly_hand(train_data, train_data, args.kernel_degree, args.kernel_gamma)
    #print(ker_train)
    ker_test = ker_poly_hand(train_data, test_data, args.kernel_degree, args.kernel_gamma)
    #print(ker_test.shape)
    if args.kernel == "rbf":
        ker_train = ker_rbf_hand(train_data, train_data, args.kernel_gamma)
        ker_test = ker_rbf_hand(train_data, test_data, args.kernel_gamma)


    train_rmses, test_rmses = [], []
    test_predictions = []
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # from primary LR:
        """
        for i in range(train_data.shape[0]//args.batch_size):
            grad_sum = np.zeros(train_data.shape[1])
            for j in range(args.batch_size):
                index = permutation[i*args.batch_size+j]
                grad = (np.transpose(train_data[index]) @ weights - train_target[index]) * train_data[index]
                grad_sum += grad
            grad_sum = grad_sum/args.batch_size
            weights = weights - args.learning_rate * (grad_sum + args.l2 * weights)
        """

        # dual LR:
        for i in range(train_data.shape[0] // args.batch_size):
            grad_betas = np.zeros(args.data_size)
            grad_bias = 0
            grad_bias2 = 0
            for j in range(args.batch_size):
                index = permutation[i * args.batch_size + j]

                mini_sum = 0
                for k in range(args.data_size):
                    mini_sum += betas[k] * (ker_train[index][k]) #- bias
                grad_betas[index] += mini_sum - train_target[index] + bias
                #print(train_target)

                #print(ker_train[index].shape, betas.shape, np.dot(ker_train[index], betas), train_target[index])
                #grad_betas[index] = np.dot(ker_train[index], betas) - train_target[index]
                #grad_betas[index] += ker_train[index].T @ betas - train_target[index]

                grad_bias += grad_betas[index] #- bias
                #grad_bias += grad_betas[index] - (args.learning_rate / args.batch_size) * bias
                #grad_bias += grad_betas[index] + (args.learning_rate / args.batch_size) * bias
                #bias -= (args.learning_rate / args.batch_size) * grad_bias
                #grad_bias += grad_betas[index]
            #grad_betas *= args.learning_rate / args.batch_size
            #bias = grad_bias



            betas -= (args.learning_rate / args.batch_size) * grad_betas
            betas -= args.learning_rate * args.l2 * betas

            for j in range(args.batch_size):
                index = permutation[i * args.batch_size + j]
                grad_bias2 += betas[index] + bias

            #bias -= (args.learning_rate / args.batch_size) * grad_bias2
            bias -= (args.learning_rate / args.batch_size) * grad_bias
            #bias -= grad_bias
            bias -= args.learning_rate * args.l2 * bias

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, compute their gradient of the loss, and
        # update the `betas` and the `bias`. You can assume that `args.batch_size`
        # exactly divides `train_data.shape[0]`.
        #
        # We assume the primary formulation of our model is
        #   y = phi(x)^T w + bias,
        # the weights are represented using betas in the dual formulation
        #   w = \sum_i beta_i phi(x_i),
        # and the loss for a batch $B$ in the primary formulation is the MSE with L2 regularization:
        #   L = \sum_{i \in B} 1/|B| * [1/2 (phi(x_i)^T w + bias - t_i)^2] + 1/2 * args.l2 * ||w||^2
        # You should update the `betas` and the `bias`, so that the update
        # is equivalent to the update in the primary formulation. Be aware that
        # for a single batch, only some betas are updated because of the MSE, but
        # all betas are updated because of L2 regularization.
        #
        # Instead of using the feature map $phi$ directly, we use a given kernel computing
        #   K(x, z) = phi(x)^T phi(z)
        # We consider the following `args.kernel`s:
        # - "poly": K(x, z; degree, gamma) = (gamma * x^T z + 1) ^ degree
        # - "rbf": K(x, z; gamma) = exp^{- gamma * ||x - z||^2}
        # The kernel parameters are specified in `args.kernel_gamma` and `args.kernel_degree`.

        # TODO: Append current RMSE on train/test data to `train_rmses`/`test_rmses`.

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