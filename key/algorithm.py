import numpy as np
from collections import Counter

# Decision Tree Node
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        # Initialize the node with the feature to split on, threshold value, left & right child nodes, and leaf value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Only set for leaf nodes (end of tree)

    def is_leaf(self):
        # Return True if the node is a leaf node (i.e., it has a value)
        return self.value is not None


# Decision Tree Classifier
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        # Initialize the tree with max depth and min samples to split
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None  # Root of the tree (initially None)

    def fit(self, X, y):
        # Train the decision tree by growing the tree from the root
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        # Recursively grow the decision tree from the current depth
        n_samples, n_features = X.shape  # Get the number of samples and features
        unique_labels = np.unique(y)  # Get unique labels in the target variable

        # Stop growing if one of the stopping criteria is met
        if (depth >= self.max_depth or len(unique_labels) == 1 or n_samples < self.min_samples_split):
            # If we reached max depth, or there's only one label, or too few samples, stop and return a leaf node
            leaf_value = self._most_common_label(y)  # Most common label is the leaf value
            return Node(value=leaf_value)

        # Otherwise, select the best feature to split on
        best_feature, best_threshold = self._best_split(X, y, n_features)
        if best_feature is None:
            # If no valid split, create a leaf with the most common label
            return Node(value=self._most_common_label(y))

        # Split data based on the best feature and threshold
        left_idx, right_idx = self._split(X[:, best_feature], best_threshold)
        # Recursively build the left and right subtrees
        left_subtree = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_subtree = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)

        # Return the current node with left and right subtrees
        return Node(best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y, n_features):
        # Find the best feature and threshold to split on by evaluating information gain
        best_gain = -1
        split_idx, split_threshold = None, None

        # Loop over all features to find the best split
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])  # Get unique values of the current feature
            for threshold in thresholds:
                # Calculate information gain for the given threshold
                gain = self._information_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain  # Track the best gain
                    split_idx = feature
                    split_threshold = threshold

        # Return the best feature and threshold for the split
        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        # Calculate the information gain based on the entropy before and after the split
        left_idx, right_idx = self._split(X_column, threshold)  # Split data
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0  # If no valid split, return 0 gain

        # Calculate entropy for the parent (before the split) and the children (after the split)
        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(y[left_idx])
        right_entropy = self._entropy(y[right_idx])

        n = len(y)
        left_weight = len(left_idx) / n
        right_weight = len(right_idx) / n
        # Return the information gain (reduction in entropy)
        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def _split(self, X_column, threshold):
        # Split the data into left and right based on the threshold
        left_idx = np.where(X_column <= threshold)[0]
        right_idx = np.where(X_column > threshold)[0]
        return left_idx, right_idx

    def _entropy(self, y):
        # Calculate the entropy (measure of disorder) of the given labels
        hist = np.bincount(y)  # Count the occurrence of each label
        probs = hist / len(y)  # Calculate probabilities
        return -np.sum([p * np.log2(p) for p in probs if p > 0])  # Compute entropy

    def _most_common_label(self, y):
        # Return the most common label in the given labels
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        # Predict the labels for a given set of data using the trained tree
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        # Recursively traverse the tree to get the predicted label for a sample
        if node.is_leaf():
            return node.value  # Return the leaf value (label)

        # If the node is not a leaf, check if the sample value is less than or greater than the threshold
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)  # Go to the left subtree
        return self._traverse_tree(x, node.right)  # Go to the right subtree


# Random Forest Classifier
class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2):
        # Initialize the Random Forest with the number of trees, max depth, and min samples to split
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []  # List to store the decision trees

    def fit(self, X, y):
        # Train the Random Forest by training multiple decision trees
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)  # Create a bootstrap sample
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)  # Train a decision tree on the sample
            self.trees.append(tree)  # Add the trained tree to the list of trees

    def _bootstrap_sample(self, X, y):
        # Generate a bootstrap sample (random sampling with replacement)
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]  # Return the sample and the corresponding labels

    def predict(self, X):
        # Make predictions using all the decision trees and take the majority vote
        tree_preds = np.array([tree.predict(X) for tree in self.trees])  # Get predictions from all trees
        tree_preds = np.swapaxes(tree_preds, 0, 1)  # Transpose the array to get predictions for each sample
        # For each sample, return the most common label (majority vote)
        return np.array([self._most_common_label(preds) for preds in tree_preds])

    def _most_common_label(self, y):
        # Return the most common label (majority vote)
        counter = Counter(y)
        return counter.most_common(1)[0][0]
# Node Class: Represents the decision tree node. It stores information about the feature to split on,
# the threshold for the split,and the leaf value if it('s a leaf node.'

# DecisionTree Class: The core decision tree logic. It recursively grows a tree by finding the best feature and
# threshold to split on. It uses information gain to evaluate splits and builds left and right subtrees.)

# RandomForest Class: A collection of decision trees. It creates multiple decision trees using bootstrap samples
# of the data and uses majority voting to make predictions.

# Each tree is trained on a different subset of the data and the random forest combines their predictions by
# taking the most common result from all trees, which typically leads to better performance and generalization
# compared to a single decision tree.