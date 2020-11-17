#!/usr/bin/env python3

import pandas as pd
import numpy as np


#------------------------
# netflix data
#------------------------

df01 = pd.read_csv("../data/netflix_titles.csv.zip")


print(list(df01.columns))
print(df01.shape)
print(df01)
print(df01.head())
missing_val_count = (df01.isnull().sum())
print(missing_val_count)

np.random.seed(0)

total_cells = np.product(df01.shape)
total_missing = missing_val_count.sum()
print(total_missing/total_cells * 100)

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
# this would drop rows where ALL columns are missing data
df01.dropna(how='all', inplace=True)

# keep rows with at least 10 columns of non-NaN values
# i.e. rows with 2+ columns of missing data are dropped
df01.dropna(thresh=10, inplace=True)
print(df01.isnull().sum())

# in this example, the missing data isn't all that critical in the grand scheme of things
# so instead of dropping ~2k rows of data, we will fill in those particualar columns with 'Unknown' 
df01.fillna({'director':'Unknown',
             'cast':'Unknown',
             'country':'Unknown',
             'date_added':'Unknown',
             'rating':'Unknown'},
             inplace=True)

df01.reset_index(inplace=True)
print(df01)


#------------------------
# ramen ratings data
#------------------------

df02 = pd.read_csv("../data/ramen-ratings.csv")
print(list(df02.columns))
print(df02.shape)
print(df02.head())
print(df02.isnull().sum())

# ~95% of the 'Top Top' column has no data, so we're just going to drop that colunn entirely
# we don't need to reset_index() after this because we're dropping a column, not rows
df02.drop(['Top Ten'], axis=1, inplace=True)
print(list(df02.columns))
print(df02)
print(df02.isnull().sum())
