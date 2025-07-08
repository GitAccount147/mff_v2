#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

# mine:
import time

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bagging", default=False, action="store_true", help="Perform bagging")
parser.add_argument("--dataset", default="breast_cancer", type=str, help="Dataset to use")  # "wine"
parser.add_argument("--feature_subsampling", default=0.5, type=float, help="What fraction of features to subsample")  # 1
parser.add_argument("--max_depth", default=3, type=int, help="Maximum decision tree depth")  # None
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=10, type=int, help="Number of trees in the forest")  # 1
# If you add more arguments, ReCodEx will keep them with your default values.


class DecisionTree:
    def __init__(self, data, target, target_range, feat_sub):
        print("I am Groot")
        self.data = data
        self.target = target
        self.target_range = target_range
        self.feat_sub = feat_sub
        self.head = Node(data, target, target_range)

    def create_tree(self, max_depth):
        time_stamp = time.time()
        self.get_children(self.head, 1, max_depth)
        print("finished with one tree: {:.2f}".format(time.time() - time_stamp))

    def get_children(self, current_node, current_depth, max_depth):
        #print("creating a new node.")
        #print(current_node.instances)
        if max_depth is None or current_depth <= max_depth:
            if (np.unique(current_node.target)).shape[0] >= 2:
                time_stamp = time.time()
                feature, c, s = current_node.find_split_point_vectorized(self.feat_sub)
                print("one call: {:.2f}".format(time.time() - time_stamp))
                #print(feature, c, s)
                if c != 0:
                    current_node.split(feature, s)
                    self.get_children(current_node.left, current_depth + 1, max_depth)
                    self.get_children(current_node.right, current_depth + 1, max_depth)
                    return None
        counts = np.zeros(self.target_range)
        for i in range(self.target_range):
            counts[i] = np.count_nonzero(current_node.target == i)
        current_node.predict_value = np.argmax(counts)

    def predict(self, sample):
        current_node = self.head
        while current_node.split_feature is not None:
            feature, s = current_node.split_feature, current_node.split_value
            if sample[feature] <= s:
                current_node = current_node.left
            else:
                current_node = current_node.right
        return current_node.predict_value

    def debug(self, current_node):
        if current_node is not None:
            #print("In a node:")
            print("instances/feature/value/majority:", current_node.instances, current_node.split_feature,
                  current_node.split_value, current_node.predict_value)
            self.debug(current_node.left)
            self.debug(current_node.right)


class Node:
    def __init__(self, data, target, target_range):
        self.data = data
        self.target = target
        self.target_range = target_range
        self.instances = data.shape[0]
        self.D = data.shape[1]
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.predict_value = None

    def get_criterion(self, target):
        counts = np.zeros(self.target_range)
        size = target.shape[0]
        for i in range(self.target_range):
            counts[i] = np.count_nonzero(target == i)
        probs = counts / size
        non_zero_probs = probs[probs != 0]
        c = - size * (np.dot(non_zero_probs, np.log(non_zero_probs)))
        return c

    def get_criterion_vectorized(self, targets):
        num = len(targets)
        sizes = np.zeros((num, 1))
        counts = np.zeros((num, self.target_range))
        for i in range(num):
            sizes[i] = targets[i].shape[0]
            for j in range(self.target_range):
                counts[i][j] = np.count_nonzero(targets[i] == j)
        probs = counts / sizes
        sizes = sizes.reshape(-1)
        probs_orig = probs
        probs[probs == 0] = 1
        log_prob = np.log(probs)
        c = - sizes * np.diagonal(probs_orig @ np.transpose(log_prob)).copy()
        return c

    def find_split_point_vectorized(self, feat_sub):
        mask = feat_sub(self.D)
        c_parent = self.get_criterion(self.target)
        targets_left = []
        targets_right = []
        for i in range(self.D):
            if mask[i]:
                feature_values = np.unique(np.transpose(self.data)[i])
                splits = [((feature_values[i] + feature_values[i + 1]) / 2) for i in range(feature_values.shape[0] - 1)]
                for m in range(len(splits)):
                    dd = np.transpose(self.data)[i]
                    ll = self.target[dd <= splits[m]]
                    rr = self.target[dd > splits[m]]
                    targets_left.append(ll)
                    targets_right.append(rr)
        c_lefts = np.array(self.get_criterion_vectorized(targets_left))
        c_rights = np.array(self.get_criterion_vectorized(targets_right))
        diffs = c_lefts + c_rights - c_parent
        best_j = np.argmin(diffs)
        best_c = diffs[best_j]

        k = 0
        best_i, best_s = 0, 0
        for i in range(self.D):
            if mask[i]:
                feature_values = np.unique(np.transpose(self.data)[i])
                splits = [((feature_values[i] + feature_values[i + 1]) / 2) for i in range(feature_values.shape[0] - 1)]
                for m in range(len(splits)):
                    if k == best_j:
                        best_i = i
                        best_s = splits[m]
                    k += 1

        return best_i, best_c, best_s

    def split(self, feature, s):
        data_left = np.array([x for x in self.data if x[feature] <= s])
        data_right = np.array([x for x in self.data if x[feature] > s])
        target_left = np.array([self.target[j] for j in range(self.instances) if self.data[j][feature] <= s])
        target_right = np.array([self.target[j] for j in range(self.instances) if self.data[j][feature] > s])
        self.left = Node(data_left, target_left, self.target_range)
        self.right = Node(data_right, target_right, self.target_range)
        self.split_feature = feature
        self.split_value = s


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Create random generators.
    generator_feature_subsampling = np.random.RandomState(args.seed)
    def subsample_features(number_of_features: int) -> np.ndarray:
        return generator_feature_subsampling.uniform(size=number_of_features) <= args.feature_subsampling

    generator_bootstrapping = np.random.RandomState(args.seed)
    def bootstrap_dataset(train_data: np.ndarray) -> np.ndarray:
        return generator_bootstrapping.choice(len(train_data), size=len(train_data), replace=True)

    if args.dataset == "digits":
        target_range = 10
    elif args.dataset == "wine":
        target_range = 3
    else:
        target_range = 2

    start_time = time.time()

    forest = []
    for i in range(args.trees):
        if args.bagging:
            mask = bootstrap_dataset(train_data)
            new_train_data = []
            new_train_target = []
            for j in range(train_data.shape[0]):
                new_train_data.append(train_data[mask[j]])
                new_train_target.append(train_target[mask[j]])
            new_train_data = np.array(new_train_data)
            new_train_target = np.array(new_train_target)
        else:
            new_train_data = train_data
            new_train_target = train_target
        tree = DecisionTree(new_train_data, new_train_target, target_range, subsample_features)
        tree.create_tree(args.max_depth)
        forest.append(tree)

    train_predict = np.zeros(train_target.shape[0])
    for i in range(train_target.shape[0]):
        votes = np.zeros(target_range)
        for j in range(args.trees):
            votes[forest[j].predict(train_data[i])] += 1
        train_predict[i] = np.argmax(votes)

    test_predict = np.zeros(test_target.shape[0])
    for i in range(test_target.shape[0]):
        votes = np.zeros(target_range)
        for j in range(args.trees):
            votes[forest[j].predict(test_data[i])] += 1
        test_predict[i] = np.argmax(votes)

    train_accuracy = sklearn.metrics.accuracy_score(train_target, train_predict)
    test_accuracy = sklearn.metrics.accuracy_score(test_target, test_predict)

    print("End time:", time.time() - start_time)



    # TODO: Create a random forest on the training data.
    #
    # Use a simplified decision tree from the `decision_tree` assignment:
    # - use `entropy` as the criterion
    # - use `max_depth` constraint, to split a node only if:
    #   - its depth is less than `args.max_depth`
    #   - the criterion is not 0 (the corresponding instance targets are not the same)
    # When splitting nodes, proceed in the depth-first order, splitting all nodes
    # in the left subtree before the nodes in right subtree.
    #
    # Additionally, implement:
    # - feature subsampling: when searching for the best split, try only
    #   a subset of features. Notably, when splitting a node (i.e., when the
    #   splitting conditions [depth, criterion != 0] are satisfied), start by
    #   generating a feature mask using
    #     subsample_features(number_of_features)
    #   which gives a boolean value for every feature, with `True` meaning the
    #   feature is used during best split search, and `False` it is not
    #   (i.e., when `feature_subsampling == 1`, all features are used).
    #
    # - train a random forest consisting of `args.trees` decision trees
    #
    # - if `args.bagging` is set, before training each decision tree
    #   create a bootstrap sample of the training data by calling
    #     dataset_indices = bootstrap_dataset(train_data)
    #   and if `args.bagging` is not set, use the original training data.
    #
    # During prediction, use voting to find the most frequent class for a given
    # input, choosing the one with the smallest class number in case of a tie.

    # TODO: Finally, measure the training and testing accuracy.
    #train_accuracy, test_accuracy = ...

    return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))