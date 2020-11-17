#!/usr/bin/env python3

import pandas as pd


#------------------------
# concat
#------------------------

# the read_csv() can read from compressed files as well
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
data01 = pd.read_csv("../data/CAvideos.csv.zip")
data02 = pd.read_csv("../data/GBvideos.csv.zip")
data03 = pd.read_csv("../data/USvideos.csv.zip")

combined_data01 = pd.concat([data01, data02, data03])

# IMPORTANT: after you combine data files, you need to reset the index
# otherwise you will have multiple rows with the same index number
# which isn't helpful when you're trying to reference the data by row
combined_data01.reset_index(drop=True, inplace=True)

print(combined_data01)
print(list(combined_data01.columns))
missing_val_count_by_column = (combined_data01.isnull().sum())
print(missing_val_count_by_column)


#------------------------
# join
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html
#------------------------

left = data01.set_index(['title', 'trending_date'])
print(left)

right = data02.set_index(['title', 'trending_date'])
print(right)

# get videos that are trending in the same day in CA and GB
# you need to provide lsuffix and rsuffix here as both data sets have the same column name
# and you're doing the join on the index
lr_joined01 = left.join(right, lsuffix='_CAN', rsuffix='_GB')
print(lr_joined01)

new_right = data03.set_index(['title', 'trending_date'])
print(new_right)

# get videos that are trending in the same day in CA and US
lr_joined02 = left.join(new_right, lsuffix='_CAN', rsuffix='_US')
print(lr_joined02)
