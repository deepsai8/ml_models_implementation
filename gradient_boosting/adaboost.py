import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))

def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)
    """
    ### BEGIN SOLUTION
    data = np.loadtxt(filename, delimiter = ',')
    X, Y = data[:, :-1], data[:,-1]
    Y = np.where(Y==0, -1, 1)
    ### END SOLUTION
    return X, Y


def adaboost(X, y, num_iter, max_depth=1):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is {-1, 1}
    """
    trees = []
    trees_weights = [] 
    N, _ = X.shape
    d = np.ones(N) / N
    ### BEGIN SOLUTION
    for m in range(num_iter):
        h = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        y_pred = h.fit(X, y, sample_weight=d).predict(X)
        err = [y != y_pred] @ d
        err += 1e-9
        alpha = 0.5 * np.log((1-err)/err)
        d = np.where(y != y_pred, np.exp(alpha), np.exp(-alpha)) * d
        d = d / np.sum(d)
        trees.append(h)
        trees_weights.append(alpha)
    ### END SOLUTION
    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y
    """
    # X input, y output
    N, _ =  X.shape
    y = np.zeros(N)
    ### BEGIN SOLUTION
    for i in range(len(trees)):
        y_pred = trees[i].predict(X) * trees_weights[i]
        y += y_pred
    ### END SOLUTION
    return np.sign(y)
