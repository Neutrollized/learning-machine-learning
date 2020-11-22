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
normalized_data = stats.boxcox(original_data)

fig, ax = plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized Data")
plt.show()
