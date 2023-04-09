import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def load_dataset(path="data/rent-ideal.csv"):
    dataset = np.loadtxt(path, delimiter=",", skiprows=1)
    y = dataset[:, -1]
    X = dataset[:, 0:- 1]
    return X, y

def gradient_boosting_mse(X, y, num_iter, max_depth=1, nu=0.1):
    """Given predictors X, an array y, and num_iter (big M in the sum)
    
    Set random_state = 0 in your DecisionTreeRegressor() trees.

    Return the y_mean and trees 
   
    Input: X, y, num_iter
           max_depth
           nu (shrinkage parameter)

    Outputs:y_mean, array of trees from DecisionTreeRegressor 
    """
    trees = []
    N, _ = X.shape
    y_mean = np.mean(y)
    fm = y_mean
    ### BEGIN SOLUTION
    for m in range(num_iter):
        res = y - fm
        t = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
        res_pred = t.fit(X, res).predict(X)
        fm_new = fm + nu*res_pred
        fm = fm_new
        trees.append(t)
    ### END SOLUTION
    return y_mean, trees  

def gradient_boosting_predict(X, trees, y_mean,  nu=0.1):
    """
    Given X, trees, y_mean predict y_hat
    """
    ### BEGIN SOLUTION
    y_hat = y_mean
    for t in range(len(trees)):
        y_hat += trees[t].predict(X) * nu
    ### END SOLUTION
    return y_hat

