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
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")  # gini
parser.add_argument("--dataset", default="digits", type=str, help="Dataset to use")  # "wine"
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")  # None
parser.add_argument("--max_leaves", default=8, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")  # 2
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")  # 42
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


class DecisionTree:
    def __init__(self, data, target, criterion, target_range):
        print("I am Groot")
        self.data = data
        self.target = target
        self.target_range = target_range
        self.head = Node(data, target, criterion, target_range)
        self.node_to_split = self.head
        self.split_par = None

    def create_tree(self, max_depth, max_leaves, min_to_split):
        if max_leaves is None:
            self.get_children(self.head, 1, max_depth, min_to_split)
        else:
            current_leaves = 2
            while self.node_to_split is not None and current_leaves <= max_leaves:
                self.node_to_split = None
                self.get_splits(self.head, 1, max_depth, min_to_split)
                if self.node_to_split is not None:
                    self.node_to_split.split(self.split_par[0], self.split_par[2])
                    current_leaves += 1

    def get_major(self, current_node):
        if current_node is not None:
            counts = np.zeros(self.target_range)
            for i in range(self.target_range):
                counts[i] = np.count_nonzero(current_node.target == i)
            current_node.predict_value = np.argmax(counts)
            self.get_major(current_node.left)
            self.get_major(current_node.right)

    def get_splits(self, current_node, current_depth, max_depth, min_to_split):
        if max_depth is None or current_depth <= max_depth:
            if min_to_split is None or current_node.instances >= min_to_split:
                if current_node.left is None:
                    feature, c, s = current_node.find_split_point()
                    if c != 0:
                        if self.node_to_split is None or c < self.split_par[1]:
                            self.node_to_split = current_node
                            self.split_par = (feature, c, s)
                else:
                    self.get_splits(current_node.left, current_depth + 1, max_depth, min_to_split)
                    self.get_splits(current_node.right, current_depth + 1, max_depth, min_to_split)


    def get_children(self, current_node, current_depth, max_depth, min_to_split):
        #print("creating a new node.")
        #print(current_node.instances)
        if max_depth is None or current_depth <= max_depth:
            if min_to_split is None or current_node.instances >= min_to_split:
                feature, c, s = current_node.find_split_point()
                #print(feature, c, s)
                if c != 0:
                    current_node.split(feature, s)
                    self.get_children(current_node.left, current_depth + 1, max_depth, min_to_split)
                    self.get_children(current_node.right, current_depth + 1, max_depth, min_to_split)
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
            print("In a node:")
            print("instances/feature/value/majority:", current_node.instances, current_node.split_feature,
                  current_node.split_value, current_node.predict_value)
            self.debug(current_node.left)
            self.debug(current_node.right)


class Node:
    def __init__(self, data, target, criterion, target_range):
        self.data = data
        self.target = target
        self.criterion = criterion
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
        if self.criterion == "gini":
            c = size * (np.dot(probs, np.ones(self.target_range) - probs))
        else:
            non_zero_probs = probs[probs != 0]
            c = - size * (np.dot(non_zero_probs, np.log(non_zero_probs)))
        return c

    def find_split_point(self):
        time_start = time.time()
        c_parent = self.get_criterion(self.target)
        points = np.zeros((2, self.D))
        for i in range(self.D):
            feature_values = np.unique(np.transpose(self.data)[i])
            splits = [((feature_values[i] + feature_values[i + 1]) / 2) for i in range(feature_values.shape[0] - 1)]
            split_max = c_parent
            best_s = 0
            for s in splits:
                target_left = np.array([self.target[j] for j in range(self.instances) if self.data[j][i] <= s])
                target_right = np.array([self.target[j] for j in range(self.instances) if self.data[j][i] > s])
                c_left = self.get_criterion(target_left)
                c_right = self.get_criterion(target_right)
                diff = c_left + c_right - c_parent
                if diff < split_max:
                    split_max = diff
                    best_s = s
            points[0][i] = split_max
            points[1][i] = best_s
        best_feature = np.argmin(points[0])
        #print("time:", time.time() - time_start)
        return best_feature, points[0][best_feature], points[1][best_feature]

    def split(self, feature, s):
        data_left = np.array([x for x in self.data if x[feature] <= s])
        data_right = np.array([x for x in self.data if x[feature] > s])
        target_left = np.array([self.target[j] for j in range(self.instances) if self.data[j][feature] <= s])
        target_right = np.array([self.target[j] for j in range(self.instances) if self.data[j][feature] > s])
        self.left = Node(data_left, target_left, self.criterion, self.target_range)
        self.right = Node(data_right, target_right, self.criterion, self.target_range)
        self.split_feature = feature
        self.split_value = s


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    timer = time.time()

    if args.dataset == "digits":
        target_range = 10
    else:
        target_range = 2

    decision_tree = DecisionTree(train_data, train_target, args.criterion, target_range)
    decision_tree.create_tree(args.max_depth, args.max_leaves, args.min_to_split)
    decision_tree.get_major(decision_tree.head)

    print("time after creation:", time.time() - timer)
    print("now i will debug:")
    decision_tree.debug(decision_tree.head)

    train_predict = np.zeros(train_target.shape[0])
    for i in range(train_target.shape[0]):
        train_predict[i] = decision_tree.predict(train_data[i])

    test_predict = np.zeros(test_target.shape[0])
    for i in range(test_target.shape[0]):
        test_predict[i] = decision_tree.predict(test_data[i])

    train_accuracy = sklearn.metrics.accuracy_score(train_target, train_predict)
    test_accuracy = sklearn.metrics.accuracy_score(test_target, test_predict)


    # TODO: Manually create a decision tree on the training data.
    #
    # - For each node, predict the most frequent class (and the one with
    #   the smallest number if there are several such classes).
    #
    # - When splitting a node, consider the features in sequential order, then
    #   for each feature consider all possible split points ordered in ascending
    #   value, and perform the first encountered split decreasing the criterion
    #   the most. Each split point is an average of two nearest unique feature values
    #   of the instances corresponding to the given node (e.g., for four instances
    #   with values 1, 7, 3, 3, the split points are 2 and 5).
    #
    # - Allow splitting a node only if:
    #   - when `args.max_depth` is not `None`, its depth must be less than `args.max_depth`;
    #     depth of the root node is zero;
    #   - there are at least `args.min_to_split` corresponding instances;
    #   - the criterion value is not zero.
    #
    # - When `args.max_leaves` is `None`, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not `None`), always split a node where the
    #   constraints are valid and the overall criterion value ($c_left + c_right - c_node$)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).

    # TODO: Finally, measure the training and testing accuracy.
    #train_accuracy, test_accuracy = ...

    return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))