# README
If you have any Jupyter notebooks (*.ipynb) files that you want to convert to other formats (i.e. python), use the [nbconvert](https://github.com/jupyter/nbconvert) tool from Jupyter.

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

`04a_melbourne_model.py` improves on this:
- by [imputing](https://en.wikipedia.org/wiki/Imputation_(statistics)) the missing data rather than dropping them entirely, which increases our training data set and produce better results (note that this will run considerably longer as you have more than 2x the data over the previous versions)
- instead of specifying a set of features manually, I'm going have it use all columns with numerical value/data types(**dtypes**)

`04b_melbourne_model.py` improves on this:
- using an **extension to imutation**, this adds columns to your data to note which rows had missing data to help your model make better predictions as the imputed values aren't actual data and would be above/below what the actual would be

`05_melbourne_model.py` improves on this:
- taking the **Date** column value of *M/DD/YYYY* and turn it into 2 columns called **Date_Month** and **Date_Year**
- this will split the categorical **Date** column into 2 numerical columns to be interpretty separately 


## NOTES
- any `NaN` in the data from printing the data or via `print(data.head())`, etc. means *Not a Number* (i.e. missing data)
- overfitting a the result of having to few weights -- leading to a limit of how much it can learn -- and then it has to base its results from this (which is biased) and could then give you the wrong results hence why **data cleaning** is so important
