#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

######################################################################################

#---------------------
# ingest data
# https://www.kaggle.com/dansbecker/melbourne-housing-snapshot/download
#---------------------
melbourne_file_path = 'data/melb_data.csv'
X_full = pd.read_csv(melbourne_file_path)

# https://stackoverflow.com/questions/55776571/how-to-split-a-date-column-into-separate-day-month-year-column-in-pandas/55776634
X_full.Date = pd.to_datetime(X_full.Date, infer_datetime_format=True)
X_full['Date_Month'] = X_full['Date'].dt.month
X_full['Date_Year'] = X_full['Date'].dt.year
X_full.drop(['Date'], axis=1, inplace=True)
print("X_full shape: {}".format(X_full.shape))
#X_full.to_csv('sample.csv', index=True)


#-----------------------------
# building learning model
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
#-----------------------------
X_full.Price.dropna(axis=0, inplace=True)
y = X_full.Price
X_full.drop(['Price'], axis=1, inplace=True)

# instead of specifying features, we're going to use all columns with numerical predictors
X = X_full.select_dtypes(exclude=['object'])

# split training/validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
print("train_X shape: {}".format(train_X.shape))

# get columns with missing values
cols_with_missing = [col for col in train_X.columns if train_X[col].isnull().any()]


#----------------------------------
# impute data (fill in the blanks)
# https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
#----------------------------------
# make copy to avoid changing origina data (when imputing)
train_X_plus = train_X.copy()
val_X_plus = val_X.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    train_X_plus[col + '_was_missing'] = train_X_plus[col].isnull()
    val_X_plus[col + '_was_missing'] = val_X_plus[col].isnull()

my_imputer = SimpleImputer()
imputed_train_X_plus = pd.DataFrame(my_imputer.fit_transform(train_X_plus))
imputed_val_X_plus = pd.DataFrame(my_imputer.transform(val_X_plus))

# Imputation removed column names; put them back
imputed_train_X_plus.columns = train_X_plus.columns
imputed_val_X_plus.columns = val_X_plus.columns
print("imputed_train_X_plus shape: {}".format(imputed_train_X_plus.shape), end='\n\n')


#------------------------------
# evaluate serveral models
#------------------------------
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

def score_model(model, tX=imputed_train_X_plus, vX=imputed_val_X_plus, ty=train_y, vy=val_y):
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
