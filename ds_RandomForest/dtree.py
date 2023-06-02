import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test[self.col] < self.split:
            return self.lchild.predict(x_test)
        if x_test[self.col] >= self.split:
            return self.rchild.predict(x_test)

    def leaf(self, x_test): #entire function added for rf, it's a walk to the leaf node
        """
        Given a single test record, x_test, return the leaf node reached by running
        it down the tree starting at this node.  This is just like prediction, 
        except we return the decision tree leaf rather than the prediction from that leaf.
        """
        if x_test[self.col] < self.split:
            return self.lchild.leaf(x_test)
        if x_test[self.col] >= self.split:
            return self.rchild.leaf(x_test)
        
        
class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction
        self.y = y # added for rf

    def predict(self, x_test):
        # return prediction
        return prediction 
    
    def leaf(self, x_test): #added for rf to return self
        return self

def gini(x):
    """
    Return the gini impurity score for values in y
    Assume y = {0,1}
    Gini = 1 - sum_i p_i^2 where p_i is the proportion of class i in y
    """
    _, counts = np.unique(x, return_counts=True) # assuming x is np.array
    p = counts / len(x)
    return 1 - np.sum( p**2 )

    
def find_best_split(X, y, loss, min_samples_leaf, max_features):
    best = {'feature': -1, 'split': -1, 'loss': loss(y)}
    cols = np.random.choice(range(len(X[0])), size = int(len(X[0])*max_features), replace = False) # previous version: range(len(X[0]))
    ncols = len(cols)
    for i in cols:
        candidates = np.random.choice(X[:,i], size = 20)
        candidates = list(set(candidates))
        for split in candidates:
            yl = y[X[:,i] < split]
            yr = y[X[:,i] >= split]
            if len(yl) < min_samples_leaf or len(yr) < min_samples_leaf:
                continue
            l = (len(yl) * loss(yl) + len(yr) * loss(yr)) / (len(yl) + len(yr))
            if l == 0:
                return i, split
            if l < best['loss']:
                best = {'feature':i, 'split':split, 'loss':l}
    return best['feature'], best['split']
    
    
class DecisionTree621:
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.var for regression or gini for classification

    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for  either a classifier or regression.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressions predict the average y
        for observations in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)

    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classification or regression.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621.create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
        if len(X) <= self.min_samples_leaf or len(np.unique(X, axis = 0))==1:
            return self.create_leaf(y)
        else:
            col,split = find_best_split(X,y,self.loss,self.min_samples_leaf, self.max_features)
            if col == -1:
                return self.create_leaf(y)
            lchild = self.fit_(X[X[:,col] < split],y[X[:,col] < split])
            rchild = self.fit_(X[X[:,col] >= split],y[X[:,col] >= split])
            return DecisionNode(col, split, lchild, rchild)

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        y_pred=[]
        for x in X_test:
            y_pred.append(self.root.predict(x))
        return np.array(y_pred)


class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1, max_features = 0.3):
        super().__init__(min_samples_leaf, loss=np.var) # also try loss=np.std
        self.max_features = max_features
        
    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        return r2_score(y_test, self.predict(X_test))

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))


class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1, max_features = 0.3):
        super().__init__(min_samples_leaf, loss=gini)
        self.max_features = max_features
        
    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        return accuracy_score(y_test, self.predict(X_test))

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor. Feel free to use scipy.stats to use the mode function.
        """
        #print(f'create leaf: {stats.mode(y)[0][0]}')
        return LeafNode(y, stats.mode(y)[0][0])
