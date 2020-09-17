# README

`00_melbourne_model.py` contains a very simple data ingestion, build model and validation

`01_melbourne_model.py` improves on this:
- using scikit's [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) split training and validation data
- `00_melbourne_model` contains training data which skews validation by reducing your error with non-training data

`02_melbourne_model.py` improves on this:
- using [Random Forests](https://en.wikipedia.org/wiki/Random_forest) to overcome some of the overfitting/underfitting issues with Decision Trees

## NOTE
- any `NaN` in the data from printing the data or via `print(data.head())`, etc. means *Not a Number* (i.e. missing data)
