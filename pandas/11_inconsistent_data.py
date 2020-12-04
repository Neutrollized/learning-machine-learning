#!/usr/bin/env python3

import pandas as pd
import numpy as np

# helpful modules
import fuzzywuzzy
from fuzzywuzzy import process
import chardet

np.random.seed(0)


#--------------------------------
# Import data 
#--------------------------------
profs = pd.read_csv("../data/pakistan_intellectual_capital.csv")

print(profs.columns)
print(profs.head())
print(profs.describe)


'''
#------------------------------------
# Fix inconsistencies (singles)
# this block is commented out
#------------------------------------
countries = profs['Country'].unique()
countries.sort()
print(countries)

# convert to lower case
profs['Country'] = profs['Country'].str.lower()
profs['Country'] = profs['Country'].str.strip()

countries = profs['Country'].unique()
countries.sort()
print(countries)

# https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
matches = fuzzywuzzy.process.extract("south korea", countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
print(matches)

countries = profs['Country'].unique()
countries.sort()
print(countries)
'''


#------------------------------------
# Fix inconsistencies (function)
#------------------------------------
# https://thispointer.com/how-to-get-check-data-types-of-dataframe-columns-in-python-pandas/
def fix_string_inconsistency(df):
  for i in range(len(df.columns)):
    column = df.columns[i]
    
    if df.dtypes[column] == np.object:
      df[column] = df[column].str.lower()
      df[column] = df[column].str.strip()

  # let us know the function's done
  print("Inconsistencies fixed!")


fix_string_inconsistency(profs)


# function to replace rows in the provided column of the provided dataframe
# that match the provided string above the provided ratio with the provided string
def replace_matches_in_column(df, column, string_to_match, min_ratio):
  # get a list of unique strings
  strings = df[column].unique()
    
  # get the top 10 closest matches to our input string
  matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                       limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

  # only get matches with a ratio > min_ratio
  close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

  # get the rows of all the close matches in our dataframe
  rows_with_matches = df[column].isin(close_matches)

  # replace all rows with close matches with the input matches 
  df.loc[rows_with_matches, column] = string_to_match
    
  # let us know the function's done
  print("All done!")


replace_matches_in_column(df=profs, column='Country', string_to_match="south korea", min_ratio=47)
replace_matches_in_column(df=profs, column='Country', string_to_match="usa", min_ratio=70)

countries = profs['Country'].unique()
countries.sort()
print(countries)
