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


#-----------------------------
# summary functions examples
#-----------------------------

print(data02.Team.describe())
print(data02.Team.unique())

print(data02.Salary.describe())
print(data02.Salary.mean())
print(data02.Salary.value_counts())


#----------------------------
# groupwise analysis
#----------------------------

# group by team name
# show count (number of entries of the team)
print(data02.groupby('Team').Team.count())

# group by team name
# show min age in the team
print(data02.groupby('Team').Age.min())

# group by team name
# print first player name on each team
print(data02.groupby('Team').apply(lambda x: x.Name.iloc[0]))

# multi-index, group by team name & position
# apply different functions to DataFrame simultaneously
# len, here applies to the length of the position list (i.e. number of players in each position for their respective teams)
data03 = data02.groupby(['Team','Position']).Salary.agg([len, min, max])
print(data03)

# you can reset indexes back to the default ones with the reset_index() method
print(data03.reset_index())

# multi-index, group by team name & position
# find the most expensive player in each position on each team
data04 = data02.groupby(['Team','Position']).apply(lambda x: x.loc[x.Salary.idxmax()])
print(data04.index)
print(data04)


#-----------------
# sorting
#-----------------

print(data02.sort_values(by='Name'))
print(data03.sort_values(by='min', ascending=False))

# can sort by multiple columns
print(data03.sort_values(by=['Team','max']))

# multi-index, group by team name & position
# find the size (number of players) in each position on each team
# sort in descending order
data05 = data02.groupby(['Team','Position']).size().sort_values(ascending=False)
print(data05)
