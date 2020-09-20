# README

## Common ML Methods
- **Regression** is a form of *supervised* machine learning that is used to predict a numeric label based on an item's features. For example, an automobile sales company might use the characteristics of car (such as engine size, number of seats, mileage, and so on) to predict its likely selling price. In this case, the characteristics of the car are the features, and the selling price is the label.
- **Classification** is a form of *supervised* machine learning that is used to predict which category, or class, an item belongs to. For example, a health clinic might use the characteristics of a patient (such as age, weight, blood pressure, and so on) to predict whether the patient is at risk of diabetes. In this case, the characteristics of the patient are the features, and the label is a classification of either 0 or 1, representing non-diabetic or diabetic.
- **Clustering** is a form of *unsupervised* machine learning that is used to group similar items into clusters based on their features. For example, a researcher might take measurements of penguins, and group them based on similarities in their proportions.


## Examples
`00_melbourne_model.py` contains a very simple data ingestion, build model and validation

`01_melbourne_model.py` improves on this:
- using scikit's [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) split training and validation data
- previously, testing data also contained training data which taints validation by reducing your error

`02_melbourne_model.py` improves on this:
- using [Random Forests](https://en.wikipedia.org/wiki/Random_forest) to overcome some of the overfitting/underfitting issues with Decision Trees

`03_melbourne_model.py` improves on this:
- by running multiple models (with various parameters tuned) to find the best result out of those

`04_melbourne_model.py` improves on this:
- imputing the missing data rather than dropping them entirely, which increases our training data set and produce better results (note that this will run considerably longer as you have more than 2x the data over the previous versions)


## NOTES
- any `NaN` in the data from printing the data or via `print(data.head())`, etc. means *Not a Number* (i.e. missing data)

