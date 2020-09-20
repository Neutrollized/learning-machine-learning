#!/usr/bin/env python3

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

######################################################################################

#---------------------
# ingest data
# https://www.kaggle.com/dansbecker/melbourne-housing-snapshot/download
#---------------------
melbourne_file_path = 'data/melb_data.csv'
X_full = pd.read_csv(melbourne_file_path)

# print Column headings
print(X_full.columns, end='\n\n')

# drop data with missing values
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
X_reduced = X_full.dropna(axis=0)

print(X_reduced.describe)
print(X_reduced.describe())

print("Before dropping data with missing values: {}".format(X_full.shape))
# Number of missing values in each column of training data
missing_val_count_by_column = (X_full.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0], end='\n\n')

print("After dropping data with missing values: {}".format(X_reduced.shape), end='\n\n')


#-----------------------------
# building learning model
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
#-----------------------------
# setting the prediction target (traditionally labeled 'y')
y = X_reduced.Price

# choosing the training features (traditionally labeled 'X')
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = X_reduced[features]

# good habit to manually inspect some of the data
X.describe()
X.head()

melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are:")
print(melbourne_model.predict(X.head()), end='\n\n')


#-----------------------------
# model validation (MAE)
# https://scikit-learn.org/stable/modules/classes.html?highlight=metrics#module-sklearn.metrics
#-----------------------------
predicted_home_prices = melbourne_model.predict(X)
MAE = mean_absolute_error(y, predicted_home_prices)
print("Mean Absolute Error in predicted home prices: " + str(round(MAE, 2)), end='\n\n')
