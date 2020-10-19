# DecoratedDecisionTree

This repository contains a machine learning regression algorithm which
adds a bit more flexibility to sklearn's `DecisionTreeRegressor` by allowing any sklearn
Regression algorithm to be fit at the leaves of the tree.

To see how this model works, see the following .

The following [example](example.ipynb) shows how you can use the `DecoratedDecisionTreeRegressor` to quickly build a decision tree and perform linear regression on each of the leaves of the tree.
