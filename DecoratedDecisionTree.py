import pandas as pd
from sklearn.base import clone

class DecoratedDecisionTreeRegressor:

    def __init__(self, dtr, decorator):
        '''
        

        Parameters
        ----------
        dtr : sklearn.tree.DecisionTreeRegressor
            Decision tree regressor
        decoration : Regessor
            Regression algorithm used to fit at the leaves of the tree.

        '''
        self.dtr = dtr
        self.decorator = decorator
        self.leaf_models = dict()
        
    
    def fit(self, df_X, y):
        '''
        Fits the decorated decision tree regressor

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
            

if __name__ == '__main__':
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LinearRegression
    ddtr = DecoratedDecisionTreeRegressor(DecisionTreeRegressor(max_depth = 2), LinearRegression())
    
    data_X = generation[['PLANT_ID', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD']]
    y = generation['TOTAL_YIELD']
    
    ddtr.fit(data_X, y)
    data_X['y'] = ddtr.predict(data_X)

