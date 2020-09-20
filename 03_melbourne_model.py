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
X_full = pd.read_csv(melbourne_file_path)

# drop data with missing values
X_reduced = X_full.dropna(axis=0)

print("X_full shape: {}".format(X_full.shape))
print("X_reduced shape: {}".format(X_reduced.shape))


#-----------------------------
# building learning model
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
#-----------------------------
# setting the prediction target (traditionally labeled 'y')
y = X_reduced.Price

# choosing the training features (traditionally labeled 'X')
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = X_reduced[features]

# split training/validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
print("train_X shape: {}".format(train_X.shape), end='\n\n')


#------------------------------
# evaluate serveral models
#------------------------------
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

def score_model(model, tX=train_X, vX=val_X, ty=train_y, vy=val_y):
    model.fit(tX, ty)
    preds = model.predict(vX)
    return mean_absolute_error(vy, preds)

model_mae_dict = {}

for i in range(len(models)):
    print("Running model_{}...".format(i+1))
    model_mae_dict[models[i]] = score_model(models[i])

# assign the model(key) of the lowest MAE(value) to melbourne_model
melbourne_model = min(model_mae_dict, key=model_mae_dict.get)

# no need to rerun the validation again as the results already stored in the dictionary...
print("Best model: {}".format(melbourne_model))
print("Mean Absolute Error in predicted home prices: {:.2f}".format(model_mae_dict[melbourne_model]))
