#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

######################################################################################

#---------------------
# ingest data
# https://www.kaggle.com/dansbecker/melbourne-housing-snapshot/download
#---------------------
melbourne_file_path = 'data/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# drop data with missing values
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
melbourne_data = melbourne_data.dropna(axis=0)


#-----------------------------
# building learning model
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
#-----------------------------
# setting the prediction target (traditionally labeled 'y')
y = melbourne_data.Price

# choosing the training features (traditionally labeled 'X')
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[features]

# split training/validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

melbourne_model = RandomForestRegressor(random_state=1)
melbourne_model.fit(train_X, train_y)


#-----------------------------
# model validation (MAE)
# https://scikit-learn.org/stable/modules/classes.html?highlight=metrics#module-sklearn.metrics
#-----------------------------
val_predictions = melbourne_model.predict(val_X)
val_MAE = mean_absolute_error(val_y, val_predictions)
print("Mean Absolute Error in predicted home prices: {:.2f}".format(val_MAE))
