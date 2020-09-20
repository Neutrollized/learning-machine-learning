#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

######################################################################################

#---------------------
# ingest data
# https://www.kaggle.com/dansbecker/melbourne-housing-snapshot/download
#---------------------
melbourne_file_path = 'data/melb_data.csv'
X_full = pd.read_csv(melbourne_file_path)

# drop data with missing values
X_reduced = X_full.dropna(axis=0)

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

# split training/validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
print("train_X shape: {}".format(train_X.shape), end='\n\n')

melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)


#-----------------------------
# model validation (MAE)
# https://scikit-learn.org/stable/modules/classes.html?highlight=metrics#module-sklearn.metrics
#-----------------------------
val_predictions = melbourne_model.predict(val_X)
val_MAE = mean_absolute_error(val_y, val_predictions)
print("Mean Absolute Error in predicted home prices: " + str(round(val_MAE, 2)))
