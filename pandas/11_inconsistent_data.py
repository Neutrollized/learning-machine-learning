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

print(profs.head())
print(profs.describe)
countries = profs['Country'].unique()
countries.sort()
print(countries)


#---------------------------
# Fix inconsistencies
#---------------------------
# convert to lower case
profs['Country'] = profs['Country'].str.lower()
profs['Country'] = profs['Country'].str.strip()

countries = profs['Country'].unique()
countries.sort()
print(countries)


# https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
#sk_matches = fuzzywuzzy.process.extract("south korea", countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
#print(sk_matches)

#usa_matches = fuzzywuzzy.process.extract("usa", countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
#print(usa_matches)

# function to replace rows in the provided column of the provided dataframe
# that match the provided string above the provided ratio with the provided string
def replace_matches_in_column(df, column, string_to_match, min_ratio = 47):
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


replace_matches_in_column(df=profs, column='Country', string_to_match="south korea")
replace_matches_in_column(df=profs, column='Country', string_to_match="usa", min_ratio=70)

countries = profs['Country'].unique()
countries.sort()
print(countries)
