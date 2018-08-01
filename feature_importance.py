import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import set_matplotlib_formats
from sklearn.metrics import accuracy_score, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

set_matplotlib_formats('retina')
plt.style.use('ggplot')

class MyRandomForestClassifier:
    def __init__(self, n_trees=50, max_features='sqrt', **kwargs):
        self.n_trees = n_trees
        self.params = kwargs
        self.params['max_features'] = max_features
        
    def fit(self, X, y):
        '''X & y must be numpy arrays
          classes must be encoded as 0, 1, 2...'''
        
        self.n_classes = len(np.unique(y))
        self.n_samples = X.shape[0]
        
        # generate bootstrap sample indices
        index_set = set(range(self.n_samples))
        self.bootstrapped_indices = [np.random.choice(self.n_samples, size=self.n_samples) 
                                     for _ in range(self.n_trees)]
        self.oob_indices = [list(index_set - set(b)) for b in self.bootstrapped_indices]
        
        # fit all the trees!
        self.estimators = [DecisionTreeClassifier(**self.params).fit(X[b],y[b]) 
                      for b in self.bootstrapped_indices]
    
    def predict_proba(self, X):
        class_probs = np.zeros((X.shape[0], self.n_classes))
        for tree in self.estimators:
            class_probs += tree.predict_proba(X)
        return class_probs / self.n_trees
    
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class MyRandomForestRegressor:
    def __init__(self, n_trees=50, max_features='sqrt', **kwargs):
        self.n_trees = n_trees
        self.params = kwargs
        self.params['max_features'] = max_features
        
    def fit(self, X, y):
        '''X & y must be numpy arrays'''
        
        self.n_classes = len(np.unique(y))
        self.n_samples = X.shape[0]
        
        # generate bootstrap sample indices
        index_set = set(range(self.n_samples))
        self.bootstrapped_indices = [np.random.choice(self.n_samples, size=self.n_samples) 
                                     for _ in range(self.n_trees)]
        self.oob_indices = [list(index_set - set(b)) for b in self.bootstrapped_indices]
        
        # fit all the trees!
        self.estimators = [DecisionTreeRegressor(**self.params).fit(X[b],y[b]) 
                      for b in self.bootstrapped_indices]
    
    def predict(self, X):
        return sum(tree.predict(X) for tree in self.estimators)/self.n_trees

    
    
def shuffle_column(X, feature_index):
    ''' 
    Parameters
    ----------
    X: numpy array
    feature_index: int
    
    Returns
    -------
    X_new: numpy array
    
    Returns a new array identical to X but
    with all the values in column feature_index
    shuffled
    '''   
    
    X_new = X.copy()
    np.random.shuffle(X_new[:,feature_index])
    return X_new    
    
def permutation_importance(model, X_test, y_test, scorer=accuracy_score):
    ''' Calculates permutation feature importance for a fitted model
    
    Parameters
    ----------
    model: anything with a predict() method
    X_test, y_test: numpy arrays of data
        unseen by model
    scorer: function. Should be a "higher is better" scoring function,
        meaning that if you want to use an error metric, you should
        multiply it by -1 first.
        ex: >> neg_mse = lambda y1, y2: -mean_squared_error(y1, y2)
            >> permutation_importance(mod, X, y, scorer=neg_mse)
    
    Returns
    -------
    feat_importances: numpy array of permutation importance
        for each feature
    
    '''
    
    feat_importances = np.zeros(X_test.shape[1])
    test_score = scorer(y_test, model.predict(X_test))
    for i in range(X_test.shape[1]):
        X_test_shuffled = shuffle_column(X_test, i)
        test_score_permuted = scorer(y_test, model.predict(X_test_shuffled))
        feat_importances[i] = test_score - test_score_permuted
    return feat_importances

def my_oob_permutation_importance(my_rf_model, X, y, scorer=accuracy_score):
    '''Calculates permutation feature importance for a fitted random forest
        model using OOB samples
        
    Parameters
    ----------
    my_rf_model: fitted model of the class MyRandomForest
    X, y: data that was used to train my_rf_model
    scorer: function. see docstring of permutation_importance above
    '''
    
    feat_importances = np.zeros(X.shape[1])
    for oob, est in zip(my_rf_model.oob_indices, my_rf_model.estimators):
        feat_importances += permutation_importance(est, X[oob], y[oob], scorer=scorer)
    return feat_importances / my_rf_model.n_trees
    

def replace_column(X, feature_index, value):
    ''' 
    Parameters
    ----------
    X: numpy array
    feature_index: int
    value: float
    
    Returns
    -------
    X_new: numpy array
    
    Returns a new array identical to X but
    with all the values in column feature_index
    replaced with value
    '''    
    X_new = X.copy()
    X_new[:,feature_index] = value
    return X_new

def partial_dependence(model, X, feature_index, classification=True):
    '''
    Parameters
    ----------
    model: fitted model
        anything with .predict()
    X: numpy array
        data the model was trained on.
    feature_index: int
        feature to calculate partial dependence for
    classification: boolean. 
        True if the model is a classifier
           (in which case, it must have .predict_proba()
        False if the model is a regressor
        
    Returns
    -------
    x_values: numpy array
        x values to plot partial dependence over
    pdp: numpy array
        partial dependence values
        
    example:
    >> x, pdp = partial_dependece(model, X_train, 3, classification=False)
    >> plt.plot(x, pdp)
    '''
    
    x_values = np.unique(X[:,feature_index])
    pdp = np.zeros(x_values.shape)
    for i, value in enumerate(x_values):
        X_new = replace_column(X, feature_index, value)
        if classification:
            y_pred_prob = model.predict_proba(X_new)[:,1]
            y_pred_prob = np.clip(y_pred_prob, 0.001, 0.999)
            y_pred = np.log(y_pred_prob / (1 - y_pred_prob))
        else:
            y_pred = model.predict(X_new)
        pdp[i] = y_pred.mean()
    return (x_values, pdp)
        
    
def partial_dependence_2d(model, X, feature_index_1, feature_index_2, classification=True):
    '''
    see docstring for partial_dependence.
    this takes two feature indices and calculates
    the joint partial dependence
    
    example:
    >> from mpl_toolkits.mplot3d import Axes3D
    >> xx1, xx2, pdp2d = partial_dependence_2d(mod, X_train, 5, 12)
    >> fig = plt.figure()
    >> ax = Axes3D(fig)
    >> ax.plot_surface(xx1, xx2, pdp2d)
    '''
    
    x1_values = np.unique(X[:,feature_index_1])
    x2_values = np.unique(X[:,feature_index_2])
    pdp = np.zeros((len(x1_values), len(x2_values)))
    for i, val1 in enumerate(x1_values):
        for j, val2 in enumerate(x2_values):
            X_new = replace_column(X, feature_index_1, val1)
            X_new = replace_column(X_new, feature_index_2, val2)
            if classification:
                y_pred_prob = model.predict_proba(X_new)[:,1]
                y_pred_prob = np.clip(y_pred_prob, 0.001, 0.999)
                y_pred = np.log(y_pred_prob / (1 - y_pred_prob))
            else:
                y_pred = model.predict(X_new)
            pdp[i,j] = y_pred.mean()
    xx1, xx2  = np.meshgrid(x1_values, x2_values, indexing='ij')
    return (xx1, xx2, pdp)

def plot_myRf_feature_importance(X_train, y_train, scorer=accuracy_score, classification=False, n_trees=50, col_names=None, show_n=None):
    '''
    ex: >> neg_mse = lambda y1, y2: -mean_squared_error(y1, y2)
                >> permutation_importance(mod, X, y, scorer=neg_mse)
    '''
    if col_names:
        col_names = col_names

    plt.figure(figsize=(8,10))
    if not(type(X_train) is np.ndarray):
        X = X_train.values
        col_names = X_train.columns.tolist()
    else:
        X = X_train
    if not(type(y_train) is np.ndarray):
        y = y_train.values
    else:
        y = y_train

    if classification:
        m = MyRandomForestClassifier(n_trees=50)
        m.fit(X, y)
        feat_importance = my_oob_permutation_importance(m, X, y, scorer=scorer)

        importance_series = pd.Series(data=feat_importance, index=col_names)
        importance_series.sort_values().plot(kind='barh')
        plt.tight_layout()
        plt.show()
    else:
        m = MyRandomForestRegressor(n_trees=50)
        m.fit(X, y)
        feat_importance = my_oob_permutation_importance(m, X, y, scorer=scorer)
        
        importance_series = pd.Series(data=feat_importance, index=col_names)
        if show_n:
            plt.figure(figsize=(12,6))
            importance_series.sort_values()[-show_n:].plot(kind='barh', width=0.35)
        else:
            plt.figure(figsize=(12,14))
            importance_series.sort_values().plot(kind='barh')
        plt.tight_layout()
        plt.show()

    return pd.DataFrame({'Features':col_names,
              'Importance':importance_series}).sort_values('Importance', ascending=False).set_index('Features')

def plot_partial_dependence(model, df, col_name, classification=False, sample=1000, sampled=True):
    idx = df.columns.tolist().index(col_name)
    if sampled:
        X = df.sample(sample).values
    else:
        X = df.values
    x, pdp = partial_dependence(model=model, X=X, feature_index=idx, classification=classification)
    plt.plot(x, pdp)
    plt.title('Partial Dependent Plot')
    plt.xlabel(col_name)
    plt.ylabel('Target Value')
    plt.show()

def plot_partial_dependence_2D(model, df, col_name, classification=False, sample=1000, sampled=True):
    idx1, idx2 = df.columns.tolist().index(col_name[0]), df.columns.tolist().index(col_name[1])
    if sampled:
        df_sampled = df.sample(sample)
        X = df_sampled.values
    else:
        X = df.values
    xx1, xx2, pdp2d = partial_dependence_2d(model, X, idx1,
                                        idx2, classification=False)
    fig = plt.figure(figsize=(7,5))
    ax = Axes3D(fig)
    ax.plot_surface(xx1, xx2, pdp2d)
    plt.title('Partial Dependent Plot between 2 Features and Target')
    plt.show()