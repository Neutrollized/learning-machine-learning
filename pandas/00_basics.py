#!/usr/bin/env python3

import pandas as pd


#---------------------
# DataFrame
#---------------------

df01 = pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
print(df01)

df02 = pd.DataFrame({'Bob': ['I like it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})
print(df02)


df03 = pd.DataFrame({'Bob': ['I like it.', 'It was awful.'],
                     'Sue': ['Pretty good.', 'Bland.']},
                    index=['Product A', 'Product B'])
print(df03)


#-------------------
# Series
#-------------------

s01 = pd.Series([1, 2, 3, 4, 5])
print(s01)

s02 = pd.Series([30, 25, 40],
                index=['2015 Sales', '2016 Sales', '2017 Sales'],
                name='Product A')
print(s02)


#----------------------------
# Reading data files
#----------------------------

file01 = pd.read_csv("../data/melb_data.csv")

# sometimes, your data files may already have a built-in index that you can use instead
# (pd will make an index for you by default, in which case you'll have 2 index columns)
# here, I say I want to use the first column from my ingested data as the index column
#file01 = pd.read_csv("../data/melb_data.csv", index_col=0)

# shape returns (num rows, num columns)
print(file01.shape)

# write to file, exclude index
file01.to_csv('output_file.csv', index=False)

# if you want the index be included in the output
#file01.to_csv('output_file.csv')
