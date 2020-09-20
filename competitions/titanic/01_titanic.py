#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC	#support vector machine
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB

######################################################################################

#-------------------
# ingest & clean
#-------------------
X_full = pd.read_csv('data/train.csv')
train_data = X_full.copy()
# printing the count for the data will let you kow if there's any missing data
# so you know what to clean
#print(train_data.count())
#print(train_data.describe())

# replace 'male' with 0 and 'female' with 1 in the 'Sex' column
train_data.Sex.replace(
  to_replace=['male', 'female'],
  value=[0, 1],
  inplace=True)

# fill the NaN ages with the mean passenger age
train_data.Age = train_data.Age.fillna(train_data.Age.mean())

# one-hot encoding
train_data = pd.get_dummies(train_data, prefix_sep='_', columns=['Embarked'])
#print(train_data.head())

#----------------------------
# build learning model
#----------------------------
train_y = train_data.Survived
train_data.drop(['Survived'], axis=1, inplace=True)

# instead of specifying features, we're going to use all columns with numerical predictors
train_X = train_data.select_dtypes(exclude=['object'])
#features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
#train_X = train_data[features]
#print(train_X.head())

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

train_model.fit(train_X, train_y)

#---------------------------
# model validation
#---------------------------
test_data = pd.read_csv('data/test.csv')
#print(test_data.count())
#print(test_data.describe())

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
test_predictions = train_model.predict(test_X)
print(test_predictions)

result = pd.DataFrame(
  {
    'PassengerId': test_data.PassengerId,
    'Survived': test_predictions
  },
  columns=['PassengerId', 'Survived']
)

result.to_csv('gender_submission.csv', index=False)
