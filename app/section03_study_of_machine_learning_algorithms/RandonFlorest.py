# %%
# importing libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import plot_tree

# %%
# importing data
base = pd.read_csv('.data/insurance.csv', keep_default_na=False)
base.head()

# %%
# dropping unnecessary columns
base = base.drop(columns=['Unnamed: 0'])
base.head()

# %%
# discovering data
base.shape

# %%
# preparing data
y = base.iloc[:, 7].values
X = base.drop(base.columns[7], axis=1).values
X

# %%
# label encoding
label_encoder = LabelEncoder()
for i in range(X.shape[1]):
    if X[:, i].dtype == 'object':
        X[:, i] = label_encoder.fit_transform(X[:, i])

# %%
# validating data
X

# %%
# splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# %%
# creating model
model = RandomForestClassifier(random_state=1, max_depth=20, max_leaf_nodes=12, n_estimators=500)

# %%
# fitting model
model.fit(X_train, y_train)

# %%
tree_index = 499
tree_to_visualize = model.estimators_[tree_index]
plt.figure(figsize=(40,20))
plot_tree(
    tree_to_visualize,
    feature_names=base.columns[:-1],
    class_names=True,
    filled=True,
    rounded=True
)
plt.show()

# %%
# predicting
predictions = model.predict(X_test)
predictions

# %%
# evaluating
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# %%
# classification report
report = classification_report(y_test, predictions)
print(report)

# %%