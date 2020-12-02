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

# i.e. 3/22/07 --> 2007-03-22
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format="%m/%d/%y")
print(landslides['date_parsed'])
print(landslides['date_parsed'].dtype)

# plot graph of the landslides by month
landslides_months = landslides['date_parsed'].dt.month
landslides_months = landslides_months.dropna()
sns.histplot(landslides_months, kde=False, bins=12)
plt.show()

# plot graph of the landslides by day of month
landslides_days = landslides['date_parsed'].dt.day
landslides_days = landslides_days.dropna()
sns.histplot(landslides_days, kde=False, bins=31)
plt.show()
