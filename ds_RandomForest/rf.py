import numpy as np
from sklearn.utils import resample

from dtree import *

class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators # number of random trees in the forest
        self.oob_score = oob_score
        self.oob_score_ = np.nan
    
    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        self.nunique = len(np.unique(y))
        self.oob_idx = []
        trees = []
        oob_idx = []
        for i in range(self.n_estimators):
            Xb, yb, idx = resample(X, y, range(len(X[:,0])), replace=True)
            oob_idx.append(list(idx))
            if self.tname == 'Classification':
                ti = ClassifierTree621()
            elif self.tname == 'Regression':
                ti = RegressionTree621()
            ti.fit(Xb, yb)
            trees.append(ti.root)
        self.trees = trees
        #print(f'oob_idx: {oob_idx}')
        
        ## OOB Calculation
        if self.oob_score:
            if self.tname == 'Classification':
                y_pred = []
                for ind, val in enumerate(X):
                    counts = {}
                    flag = False
                    for tn, t in enumerate(trees):
                        if ind not in oob_idx[tn]:
                            flag = True
                            xleaf = t.leaf(val) #leaves reached by x
                            for j in xleaf.y:
                                if j in counts:
                                    counts[j] += 1
                                else:
                                    counts[j] = 1
                    if flag:
                        y_pred.append(sorted(counts, key= lambda x: counts[x], reverse=True)[0])
                    else:
                        y_pred.append(0)
                self.oob_score_ = accuracy_score(y, y_pred)
                
            elif self.tname == 'Regression':
                y_pred = []
                for ind, val in enumerate(X):
                    leaves = set()
                    nobs, ysum = 0, []
                    flag = False
                    for tn, t in enumerate(trees):
                        if ind not in oob_idx[tn]:
                            flag = True
                            xleaf = t.leaf(val).y #leaves reached by x
                            nobs += len(xleaf)
                            ysum = ysum + (list(xleaf))
                            
                    if flag:
                        y_pred.append(sum(ysum) / nobs)
                    else:
                        y_pred.append(0)
                
                self.oob_score_ = r2_score(y, y_pred)
            
            
class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.tname = 'Regression'
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of observations in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        y_pred=[]
        for x in X_test:
            nobs, ysum = 0, []
            for i in self.trees:
                xleaf = i.leaf(x).y #set of leaves reached by x
                nobs += len(xleaf)
                ysum = ysum + list(xleaf)
            #print(f'ysum: {ysum}')
            y_pred.append(sum(ysum) / nobs)

        return np.array(y_pred)
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        
        return r2_score(y_test, self.predict(X_test))

        
class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        n_estimators = n_estimators
        self.tname = 'Classification'
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        
    def predict(self, X_test) -> np.ndarray:
        #print(self.nunique)
        
        y_pred=[]
        for x in X_test:
            cts = []
            counts = {}
            for i in self.trees:
                xleaf = i.leaf(x) #leaves reached by x
                for j in xleaf.y:
                    if j in counts:
                        counts[j] += 1
                    else:
                        counts[j] = 1
            y_pred.append(sorted(counts, key= lambda x: counts[x], reverse=True)[0])
            #y_pred.append(stats.mode(cts))
        print(y_pred)
        return np.array(y_pred)
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        return accuracy_score(y_test, self.predict(X_test))  
