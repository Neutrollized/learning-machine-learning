#!/usr/bin/env python3

import pandas as pd
from sklearn.preprocessing import LabelEncoder
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

X_full.Date = pd.to_datetime(X_full.Date, infer_datetime_format=True)
X_full['Date_Month'] = X_full['Date'].dt.month
X_full['Date_Year'] = X_full['Date'].dt.year
X_full.drop(['Date'], axis=1, inplace=True)
print("X_full shape: {}".format(X_full.shape))


#-----------------------------
# building learning model
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
#-----------------------------
X_full.Price.dropna(axis=0, inplace=True)
y = X_full.Price
X_full.drop(['Price'], axis=1, inplace=True)

# split training/validation data
train_X, val_X, train_y, val_y = train_test_split(X_full, y, random_state=0)
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

'''
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

'''
#--------------------------------
# label encoding
#--------------------------------
#label_train_X = imputed_train_X_plus.copy()
#label_val_X = imputed_val_X_plus.copy()
label_train_X = train_X.copy()
label_val_X = val_X.copy()

# All categorical columns
object_cols = [col for col in train_X.columns if train_X[col].dtype == "object"]

# Columns that can be safely label encoded
good_label_cols = [col for col in object_cols if 
                   set(train_X[col]) == set(val_X[col])]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))
        
print('Categorical columns that will be label encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)

label_train_X = train_X.drop(bad_label_cols, axis=1)
label_val_X = val_X.drop(bad_label_cols, axis=1)

# Get list of categorical variables
#s = (label_train_X_plus.dtypes == 'object')
s = (train_X.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)

my_label_encoder = LabelEncoder()
for col in object_cols:
    #label_train_X[col] = my_label_encoder.fit_transform(imputed_train_X_plus[col])
    #label_val_X[col] = my_label_encoder.transform(imputed_val_X_plus[col])
    label_train_X[col] = my_label_encoder.fit_transform(train_X[col])
    label_val_X[col] = my_label_encoder.transform(val_X[col])


#------------------------------
# evaluate serveral models
#------------------------------
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

#def score_model(model, tX=label_train_X, vX=label_val_X, ty=train_y, vy=val_y):
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
