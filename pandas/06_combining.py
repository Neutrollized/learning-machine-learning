#!/usr/bin/env python3

import pandas as pd


#--------------------------------------
# reading & combining the data
#--------------------------------------

data01 = pd.read_csv("../data/CAvideos.csv")
data02 = pd.read_csv("../data/GBvideos.csv")
data03 = pd.read_csv("../data/USvideos.csv")

combined_data01 = pd.concat([data01, data02, data03])

# IMPORTANT: after you combine data files, you need to reset the index
# otherwise you will have multiple rows with the same index number
# which isn't helpful when you're trying to reference the data by row
combined_data01.reset_index(drop=True, inplace=True)

print(combined_data01)
print(list(combined_data01.columns))
missing_val_count_by_column = (combined_data01.isnull().sum())
print(missing_val_count_by_column)

