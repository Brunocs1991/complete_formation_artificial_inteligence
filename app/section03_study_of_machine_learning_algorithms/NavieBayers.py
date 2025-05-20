# %%

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
# %%
# import data
base = pd.read_csv('.data/insurance.csv', keep_default_na=False)
base.head()
# %%
# drop unescessary columns
base = base.drop(columns=['Unnamed: 0'])
base.head()
# %%
# discover
base.shape
# %%
# check for null values
print(base.isnull().sum())
# %%
y = base.iloc[:, 7].values
X = base.drop(base.columns[7], axis=1).values
X
# %%
# transform categorical variables to numerical values using label encoding
# Note: This is a simple approach and may not be suitable for all datasets
label_encoder = LabelEncoder()
for i in range(X.shape[1]):
    if X[:, i].dtype == 'object':
        X[:, i] = label_encoder.fit_transform(X[:, i])
# %%
# valid data
X
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=12)
# %%
# create the model
# Note: GaussianNB is a Naive Bayes classifier for Gaussian-distributed data
model = GaussianNB()
model.fit(X_train, y_train)
# %%
# predict
# Note: The predict method returns the predicted class labels for the input samples
prediction = model.predict(X_test)
# %%
# evaluate the model
prediction
# %%
accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction, average='weighted')
recall = recall_score(y_test, prediction, average='weighted')
f1 = f1_score(y_test, prediction, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
# %%
report = classification_report(y_test, prediction)
print(report)
# %%
