#!/usr/bin/env python3

import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# set seed for reproducibility
np.random.seed(0)


#------------------------
# scaling
#------------------------

original_data = np.random.exponential(size=1000)
#print(original_data)

# http://rasbt.github.io/mlxtend/user_guide/preprocessing/minmax_scaling/
scaled_data = minmax_scaling(original_data, columns=[0])

fig, ax = plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled Data")
plt.show()


#------------------------
# normalization
#------------------------

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
# NOTE: Box-Cox only takes positive values
normalized_data = stats.boxcox(original_data)

fig, ax = plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized Data")
plt.show()


#-------------------------------
# kickstarter projects example
#-------------------------------

ks_data01 = pd.read_csv("../data/ks-projects-201801.csv.zip")
print(list(ks_data01.columns))
print(ks_data01.shape)
print(ks_data01)
print(ks_data01.head())
print(ks_data01.isnull().sum())

# select the usd_goal_real column
original_data = pd.DataFrame(ks_data01.usd_goal_real)

# scale the goals from 0 to 1
scaled_data = minmax_scaling(original_data, columns=['usd_goal_real'])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2,figsize=(15,3))
sns.distplot(ks_data01.usd_goal_real, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
plt.show()

print('Original data\nPreview:\n', original_data.head())
print('Minimum value:', float(original_data.min()),
      '\nMaximum value:', float(original_data.max()))
print('_'*30)

print('\nScaled data\nPreview:\n', scaled_data.head())
print('Minimum value:', float(scaled_data.min()),
      '\nMaximum value:', float(scaled_data.max()))


# get the index of all positive pledges (Box-Cox only takes positive values)
index_of_positive_pledges = ks_data01.usd_pledged_real > 0

# get only positive pledges (using their indexes)
positive_pledges = ks_data01.usd_pledged_real.loc[index_of_positive_pledges]

# normalize the pledges (w/ Box-Cox)
normalized_pledges = pd.Series(stats.boxcox(positive_pledges)[0], 
                               name='usd_pledged_real', index=positive_pledges.index)

# plot both together to compare
fig, ax=plt.subplots(1,2,figsize=(15,3))
sns.distplot(positive_pledges, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax=ax[1])
ax[1].set_title("Normalized data")
plt.show()

print('Original data\nPreview:\n', positive_pledges.head())
print('Minimum value:', float(positive_pledges.min()),
      '\nMaximum value:', float(positive_pledges.max()))
print('_'*30)

print('\nNormalized data\nPreview:\n', normalized_pledges.head())
print('Minimum value:', float(normalized_pledges.min()),
      '\nMaximum value:', float(normalized_pledges.max()))
