# %%
# import
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# load data
mtcars = pd.read_csv('.data/mt_cars.csv')
mtcars.head()

# %%
# split data
X = mtcars[['mpg', 'hp']]
y = mtcars['cyl']

# %%
# create the model
knn = KNeighborsClassifier(n_neighbors=3)
model = knn.fit(X, y)
model

# %%
# make predictions
y_pred = model.predict(X)
y_pred

# %%    
# evaluate the model
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')


print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')


# %%
# visualize confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# %%
# other example
# 'mpg', 'hp'
new_data = pd.DataFrame([[19.3, 105]], columns=['mpg', 'hp'])
prediction = model.predict(new_data)
print(f'Predicted class for new data {new_data.values} is: {prediction[0]}')
# cyl

# %%
# visualize decision boundary
distance, indices = model.kneighbors(new_data)
print(f'Distance and indices of the nearest neighbors: {distance}, {indices}')

mtcars.loc[[1, 5 , 31], ['mpg', 'hp', 'cyl']]

# %%