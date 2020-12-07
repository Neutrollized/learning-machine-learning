#!/usr/bin/env python3

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

######################################################################################

#---------------------
# ingest data
# https://www.kaggle.com/dansbecker/melbourne-housing-snapshot/download
#---------------------
melbourne_file_path = 'data/melb_data.csv'
X_full = pd.read_csv(melbourne_file_path)

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['Price'], inplace=True)
y = X_full.Price
X_full.drop(['Price'], axis=1, inplace=True)

# instead of specifying features, we're going to use all columns with numerical predictors
X = X_full.select_dtypes(exclude=['object'])

# split training/validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
print("train_X shape: {}".format(train_X.shape))

print(X_full.head())

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in train_X.columns if
                    train_X[cname].nunique() < 10 and 
                    train_X[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in train_X.columns if 
                  train_X[cname].dtype in ['int64', 'float64']]


#-------------------
# preprocessing
#-------------------
numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='most_frequent')),
  ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
  ('num', numerical_transformer, numerical_cols),
  ('cat', categorical_transformer, categorical_cols)
])


#---------------------------------
# define the model
# evaluate pipeline
# cross-validate
#---------------------------------
my_pipeline = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('model', RandomForestRegressor(n_estimators=100, random_state=0))
])

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
# multiply by -1 since sklearn calculates negative MAE
# https://scikit-learn.org/stable/modules/model_evaluation.html
scores = -1 * cross_val_score(my_pipeline, X, y,
                               cv=5,
                               scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
print("Average MAE score (across experiments):", scores.mean())
