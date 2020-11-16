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


#-----------------------------
# summary functions examples
#-----------------------------

print(data02.Team.describe())
print(data02.Team.unique())

print(data02.Salary.describe())
print(data02.Salary.mean())
print(data02.Salary.value_counts())

# find how many players have 'Jeff' or 'Jordan' in their name
# and prints the total count (False if they don't; True if they do)
# note that you need to specify str.contains as it's a strings function
print(data02.Name.str.contains('Jeff|Jordan').value_counts())

# if you want *only* the True count, then call the sum() function
print(data02.Name.str.contains('Jeff|Jordan').sum())

# if you want *only* the False, then you'll have to subtract the "sum()" from the total 
print(len(data02.Name) - data02.Name.str.contains('Jeff|Jordan').sum())


#----------------------------
# maps using map()
# typically uses lambda()
# returns new Series
#----------------------------

print(data02.Salary.map(lambda p: p - data02.Salary.mean()))


#-------------------------------------------------
# maps using apply()
# for using custom methods applied to each row
# returns new DataFrame
#-------------------------------------------------

def remean_salary(row):
  row.Salary = row.Salary - data02.Salary.mean()
  return row

# if you had a function that applied changes to each column
# then you would have to call axis='index'
print(data02.apply(remean_salary, axis='columns'))


# we're calculating the ratio between 2 columns in all the rows
def salary_age_ratio(row):
  SAR = row.Salary / row.Age
  return SAR

data03 = data02.apply(salary_age_ratio, axis='columns')

# returns the info of the players with the highest and lowest salary-to-age ratios
# by utilizing the idxmax() and idxmin() methods respectively
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.idxmax.html
max_sar_index = data03.idxmax()
print(data02.loc[max_sar_index])

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.idxmin.html
min_sar_index = data03.idxmin()
print(data02.loc[min_sar_index])

# adding the column to the end
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.assign.html
data04 = data02.assign(Salary_Age_Ratio = data03.values)
print(data04)

# alternatively, you can add the column using this shorter method instead
# which adds the values as a new column as the custom map is being applied
data05 = data02.assign(Salary_Age_Ratio=salary_age_ratio)
print(data05)
