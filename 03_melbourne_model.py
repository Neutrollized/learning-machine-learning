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

#------------------------------
# evaluate serveral models
#------------------------------
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

def score_model(model, X_t=train_X, X_v=val_X, y_t=train_y, y_v=val_y):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

model_mae_dict = {}

for i in range(len(models)):
    #mae = score_model(models[i])
    #print("model_{} MAE: {:.2f}".format(i+1, mae))
    model_mae_dict[models[i]] = score_model(models[i])

# assign the model(key) of the lowest MAE(value) to melbourne_model
melbourne_model = min(model_mae_dict, key=model_mae_dict.get)

# no need to rerun the validation again as the results already stored in the dictionary...
print("Best model: {}".format(melbourne_model))
print("Mean Absolute Error in predicted home prices: {:.2f}".format(model_mae_dict[melbourne_model]))
