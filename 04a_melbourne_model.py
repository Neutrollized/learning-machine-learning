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
melbourne_data = pd.read_csv(melbourne_file_path)
'''
COMMENT BLOCK - we're not going to drop data this time
print("Pre-drop data count: {}".format(melbourne_data.count()))

melbourne_data = melbourne_data.dropna(axis=0)
# post-drop, you can see over half the rows are gone
print("Post-drop data count: {}".format(melbourne_data.count()))
COMMENT BLOCK END
'''


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

# get columns with missing values
cols_with_missing = [col for col in train_X.columns if train_X[col].isnull().any()]
#reduced_train_X = train_X.drop(cols_with_missing, axis=1)
#reduced_val_X = val_X.drop(cols_with_missing, axis=1)


#----------------------------------
# impute data (fill in the blanks)
# https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
#----------------------------------
my_imputer = SimpleImputer()
imputed_train_X = pd.DataFrame(my_imputer.fit_transform(train_X))
imputed_val_X = pd.DataFrame(my_imputer.transform(val_X))


#------------------------------
# evaluate serveral models
#------------------------------
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

def score_model(model, X_t=imputed_train_X, X_v=imputed_val_X, y_t=train_y, y_v=val_y):
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
