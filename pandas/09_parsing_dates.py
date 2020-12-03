#!/usr/bin/env python3

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

np.random.seed(0)


landslides = pd.read_csv("../data/landslides_data.csv")

print(landslides.columns)
print(landslides.shape)
print(landslides)
print(landslides.isnull().sum())

print(landslides.groupby('country_name').country_name.count())

print(landslides['date'].head())
print(landslides['date'].dtype)

#------------------------
# Handling anomalies
#------------------------
# one common issue is dates are entered in different formats and hence different entry lengths
# so it's a good practice to check for string lengths
# in our case, lenghts of 6, 7, and 8 are all valid (don't forget we're counting the '/' as well) 
date_lengths = landslides.date.str.len()
print(date_lengths.value_counts())

# you'll need to find out where the anomalies lie if you need/want to use a provide a different formatting string
# you can uncomment out the following piece of code if you want to find that out
#indices = np.where([date_lengths == 6])[1]
#print('Indices with anomaly data:', indices)
#landslides.loc[indices]

# and if there aren't a lot of anomalies and you want to fix it manually, you can do something like:
# landslides.[INDEX, 'date'] = 'MM/DD/YY'
# the above line would update the entry at INDEX for 'date' and set it to the MM/DD/YY format so that it's like the others 
#------------------------

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html
# i.e. 3/22/07 --> 2007-03-22
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format="%m/%d/%y")
print(landslides['date_parsed'])
print(landslides['date_parsed'].dtype)

# plot graph of the landslides by month
landslides_months = landslides['date_parsed'].dt.month
landslides_months = landslides_months.dropna()
sns.distplot(landslides_months, kde=False, bins=12)
plt.show()

# plot graph of the landslides by day of month
landslides_days = landslides['date_parsed'].dt.day
landslides_days = landslides_days.dropna()
sns.distplot(landslides_days, kde=False, bins=31)
plt.show()
