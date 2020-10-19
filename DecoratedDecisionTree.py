import pandas as pd
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')


class DecoratedDecisionTreeRegressor:

    def __init__(self, dtr, decorator):
        '''
        Creates a decorated decision tree regressor.  A decision tree is fit 
        according to the supplied DecisionTreeRegressor.  The data on the
        leaves of the tree are fit according to a supplied decorator
        which is a regression algorithm.

        Parameters
        ----------
        dtr : sklearn.tree.DecisionTreeRegressor
            Decision tree regressor
        decorator : Regessor
            Regression algorithm used to fit the data at the leaves of the tree.

        '''
        self.dtr = dtr
        self.decorator = decorator
        self.leaf_models = dict()
        
    
    def fit(self, df_X, y):
        '''
        Fits the decorated decision tree regressor.

        Parameters
        ----------
        df : DataFrame
            DataFrame containing the features we want to use for prediction.
        y : Series
            Values we are trying to predict.

        Returns
        -------
        None.

        '''
        df_X_copy = df_X.copy()
        self.dtr.fit(df_X_copy, y)
        leaves = self.dtr.apply(df_X_copy)
        # Loop over the leaves and fit the decoration regression algorithm
        # and save the result to a dictionary
        for leaf in set(leaves):
            df_X_leaf = df_X_copy[leaves==leaf]
            y_leaf = y[leaves==leaf]
            leaf_model = clone(self.decorator)
            leaf_model.fit(df_X_leaf, y_leaf)
            self.leaf_models[leaf] = leaf_model
            
    def predict(self, df_X):
        '''

        Parameters
        ----------
        df_X : DataFrame
            DataFrame containing the features used to train the model

        Returns
        -------
        Series
            A series containing the prediction.

        '''
        df_X_copy = df_X.copy()
        leaves = self.dtr.apply(df_X_copy)
        # Say what the ordering is so that we can get the same order back
        # when we are done predicting
        columns = list(df_X_copy)
        df_X_copy['__ordering'] = range(len(df_X_copy))
        df_out = pd.DataFrame({})
        # Go through the leaves and predict using the model associated to
        # the leaf
        for leaf in set(leaves):
            df_X_leaf = df_X_copy[leaves==leaf]
            model = self.leaf_models[leaf]
            df_X_leaf['y'] = model.predict(df_X_leaf[columns])
            df_out = pd.concat((df_out, df_X_leaf))
        df_out = df_out.sort_values('__ordering', ascending=True)
        return df_out['y']

