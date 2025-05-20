# %%
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
import seaborn as sns

# %%
# read the dataset
base = pd.read_csv('.data/mt_cars.csv')
base.shape

# %%
# check the first rows
base.head()

# %%
#  drop the first column
# The first column is an index column that is not needed for the analysis.
base = base.drop(columns=['Unnamed: 0'], axis=1)
base.head()

# %%
corr = base.corr()
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')


# %%
columns_pairs = [
    ['mpg', 'cyl'],
    ['mpg', 'disp'],
    ['mpg', 'hp'],
    ['mpg', 'wt'],
    ['mpg', 'drat'],
    ['mpg', 'vs']
]
n_plots = len(columns_pairs)
fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(6, 4 * n_plots))
for i, pair in enumerate(columns_pairs):
    x_col, y_col = pair
    sns.scatterplot(x=x_col, y=y_col, data=base, ax=axes[i])
    axes[i].set_title(f'Scatter plot of {x_col} vs {y_col}')

plt.tight_layout()
plt.show()

# %%
