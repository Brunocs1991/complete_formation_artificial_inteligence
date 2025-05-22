# %%
#imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
# %%
# import data
base = pd.read_csv('.data/insurance.csv', keep_default_na=False)
base.head()

# %%
# drop unnecessary columns
base = base.drop(columns=['Unnamed: 0'])
base.head()

# %%
# discover
base.shape

#%%
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
# valid data
X

# %%
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# %%
# create model
model = DecisionTreeClassifier(random_state=12, max_depth=8, max_leaf_nodes=6)

# %%
# fit model
model.fit(X_train, y_train)
plt.figure(figsize=(40,20))
plot_tree(
    model,
    feature_names=base.columns[:-1],
    class_names=True,
    filled=True,
    rounded=True
)
plt.show()

# %%
# predict
y_pred = model.predict(X_test)
y_pred

# %%
# evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# %%
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
# %%

report = classification_report(y_test, y_pred)
print(report)

# %%