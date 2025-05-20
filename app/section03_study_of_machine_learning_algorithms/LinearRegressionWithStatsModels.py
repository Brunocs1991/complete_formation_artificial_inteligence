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
# AIC:	156.6 | BIC:	162.5
# model = sm.ols(formula='mpg ~ wt + disp + hp', data=base)

# AIC:	165.1 | BIC:	169.5
# model = sm.ols(formula='mpg ~ disp + cyl', data=base)

# AIC:	179.1 | BIC:	183.5
model = sm.ols(formula='mpg ~ drat + vs', data=base)
model = model.fit()
model.summary()

# %%
residuals = model.resid
plt.hist(residuals, bins=20)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Histogram')

# %%
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Normal Q-Q plot of Residuals')
plt.show()

# %%

# h0: residuals are normally distributed
# p <= 0.05: reject the null hypothesis (not normally distributed)
# p > 0.05: fail to reject the null hypothesis (normally distributed)
stat, pval = stats.shapiro(residuals)
print(f'Shapiro-Wilk test statistic: {stat:.3f}, p-value: {pval:.3f}')

# %%
