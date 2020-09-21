#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC	#support vector machine
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

######################################################################################

#-------------------
# ingest & clean
#-------------------
train_full = pd.read_csv('data/train.csv')
train_full_plus = train_full.copy()
print("train_full shape: {}".format(train_full.shape))

# replace 'male' with 0 and 'female' with 1 in the 'Sex' column
train_full_plus.Sex.replace(
  to_replace=['male', 'female'],
  value=[0, 1],
  inplace=True)

# get columns with missing values
cols_with_missing = [col for col in train_full_plus.columns if train_full_plus[col].isnull().any()]

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    train_full_plus[col + '_was_missing'] = train_full_plus[col].isnull()

# fill the NaN ages with the mean passenger age
#train_data.Age = train_data.Age.fillna(train_data.Age.mean())

# one-hot encoding
train_data = pd.get_dummies(train_full_plus, prefix_sep='_', columns=['Embarked'])
#print(train_data.head())


#----------------------------
# build learning model
#----------------------------
train_y = train_full_plus.Survived
train_full_plus.drop(['Survived'], axis=1, inplace=True)

# instead of specifying features, we're going to use all columns with numerical predictors
train_X = train_full_plus.select_dtypes(exclude=['object'])
#features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
#train_X = train_data[features]
#print(train_X.head())

my_imputer = SimpleImputer()
imputed_train_X = pd.DataFrame(my_imputer.fit_transform(train_X))
# Imputation removed column names; put them back
imputed_train_X.columns = train_X.columns
print("imputed_train_X shape: {}".format(imputed_train_X.shape), end='\n\n')

# gotta try them all to see which works best :|
#train_model = RandomForestClassifier()
#train_model = ExtraTreesClassifier()
#train_model = AdaBoostClassifier()
train_model = GradientBoostingClassifier()
#train_model = LogisticRegression()
#train_model = KNeighborsClassifier()
#train_model = DecisionTreeClassifier()
#train_model = SVC()
#train_model = LinearSVC()
#train_model = GaussianProcessClassifier()
#train_model = GaussianNB()
#train_model = XGBClassifier()	#errors out...? :(

train_model.fit(imputed_train_X, train_y)

#---------------------------
# model validation
#---------------------------
test_data = pd.read_csv('data/test.csv')

# replace 'male' with 0 and 'female' with 1 in the 'Sex' column
test_data.Sex.replace(
  to_replace=['male', 'female'],
  value=[0, 1],
  inplace=True)

test_data.Age = test_data.Age.fillna(test_data.Age.mean())
test_data.Fare = test_data.Fare.fillna(test_data.Fare.mean())
test_data = pd.get_dummies(test_data, prefix_sep='_', columns=['Embarked'])

test_X = test_data.select_dtypes(exclude=['object'])
#test_X = test_data[features]
#print(test_X.count())
my_imputer = SimpleImputer()
imputed_test_X = pd.DataFrame(my_imputer.fit_transform(test_X))
# Imputation removed column names; put them back
imputed_test_X.columns = test_X.columns
print("imputed_test_X shape: {}".format(imputed_test_X.shape), end='\n\n')

test_predictions = train_model.predict(imputed_test_X)
print(test_predictions)

result = pd.DataFrame(
  {
    'PassengerId': test_data.PassengerId,
    'Survived': test_predictions
  },
  columns=['PassengerId', 'Survived']
)

result.to_csv('gender_submission.csv', index=False)
