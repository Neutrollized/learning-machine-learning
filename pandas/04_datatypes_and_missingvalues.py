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

# shape after dropping rows with missing values
print(data02.shape)
print(data02.head())


#------------------------
# data types
#------------------------


print(data02.Position.dtype)
print(data02.dtypes)
print(data02.index.dtype)

# you can print the salary as an int rather than float
print(data02.Salary.dtype)
print(data02.Salary.astype('int64'))

# the following option disables the "SettingWithCopyWarning"
# https://www.dataquest.io/blog/settingwithcopywarning/
pd.set_option('mode.chained_assignment', None)
data02.loc[:, 'Salary'] = data02.Salary.astype('int64')
print(data02)


#----------------------
# missing data
#----------------------

# create a DataFrame that contains all the rows where salary is missing data
data03 = data01[pd.isnull(data01.Salary)]
print(data03)

# you can fill the missing data with a value
data04 = data03.fillna('Unknown')
print(data04)

# the replace() method is very broad, like a meat cleaver
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
data04.Team.replace('Memphis Grizzlies', 'Vancouver Grizzlies', inplace=True)

# update the team column value for a particular player
# if you need a scalpel, make use of loc instead
# https://www.kite.com/python/answers/how-to-change-values-in-a-pandas-dataframe-column-based-on-a-condition-in-python
data04.loc[data04.Name == 'John Holland', "Team"] = "Chicago Bulls"
print(data04.head())
