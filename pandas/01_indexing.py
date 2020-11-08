#!/usr/bin/env python3

import pandas as pd


#-------------------------------
# reading the data
#-------------------------------

file01 = pd.read_csv("https://media.geeksforgeeks.org/wp-content/uploads/nba.csv")

# print column headings
# print(list(file01)) also works, but the following is more explicit
print(list(file01.columns))

# initial shape
print(file01.shape)

# print columns headings and number of missing values in each heading
missing_val_count_by_column = (file01.isnull().sum())
print(missing_val_count_by_column)

# drop rows with missing values
file01 = file01.dropna(axis=0)

# shape after dropping rows with missing values
print(file01.shape)


#-----------------------------
# python indexing
# column 1st, row 2nd
#-----------------------------

print(file01['Salary'])


#----------------------------------------
# pandas index-based selection
# row 1st, column 2nd
#----------------------------------------

print(file01.iloc[0])

# so if you wanted all the 4th column values...
print(file01.iloc[:, 3])

# or just rows 11-20 of the 6th column...
print(file01.iloc[10:20, 5])

# or last 5 rows of the 7-9th column...
print(file01.iloc[-5:, [6, 7, 8]])


#----------------------------------------------
# pandas attribute/label-based selection
# row 1st, column 2nd
#----------------------------------------------

print(file01.loc[0, 'College'])

print(file01.loc[:, ['Name', 'Team', 'College']])


#-----------------------------
# manipulating index
#-----------------------------

# index is now based on player name (rather than numerical/row number)
# you have 1 less column now as well
data01 = file01.set_index('Name')
print(data01)
print(data01.shape)

# using conditionals to print whether player is a PG
print(data01.Position == 'PG')

# combining multiple conditionals to print PGs that weight 200Lb+
print(data01.loc[(data01.Position == 'PG') & (data01.Weight >= 200)])
