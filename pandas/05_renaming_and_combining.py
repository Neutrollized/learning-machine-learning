#!/usr/bin/env python3

import pandas as pd


#-------------------------------
# reading the data
#-------------------------------

data01 = pd.read_csv("https://media.geeksforgeeks.org/wp-content/uploads/nba.csv")

# print column headings
# print(list(data01)) also works, but the following is more explicit
print(list(data01.columns))

# initial shape
print(data01.shape)

# print columns headings and number of missing values in each heading
missing_val_count_by_column = (data01.isnull().sum())
print(missing_val_count_by_column)

# drop rows with missing values
data02 = data01.dropna(axis=0)
data02.reset_index(drop=True, inplace=True)

# shape after dropping rows with missing values
print(data02.shape)
print(data02.head())


#-------------------
# renaming
#-------------------

data03 = data02.rename(columns={'Number': 'Jersey'})
print(data03)


#-------------------
# combining
#-------------------
