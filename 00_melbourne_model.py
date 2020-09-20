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
melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.columns, end='\n\n')

# drop data with missing values
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
melbourne_data = melbourne_data.dropna(axis=0)

print(melbourne_data.describe)
print(melbourne_data.describe())


#-----------------------------
# building learning model
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
#-----------------------------
# setting the prediction target (traditionally labeled 'y')
y = melbourne_data.Price

# choosing the training features (traditionally labeled 'X')
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[features]

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
