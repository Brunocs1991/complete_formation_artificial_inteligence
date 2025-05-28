# %%
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib.pyplot as plt


# Load the iris dataset
iris = load_iris()
iris
# %%
# function plot clyste

def pltot_clusters(data, labels, title):
    colors = ['red', 'green', 'purple', 'black']
    plt.figure(figsize=(8, 4))
    for i, c, l in zip(range(-1, 3), colors, ['Noise', 'Setosa', 'Versicolor', 'Virginica']):
        if i == -1:
            plt.scatter(data[labels == i, 0], data[labels == i, 3],
                        c=colors[i], label=l, alpha=0.5, s=50, marker='x')
        else:
            plt.scatter(data[labels == i, 0], data[labels == i, 3],
                        c=colors[i], label=l, alpha=0.5, s=50)
    plt.legend()
    plt.title(title)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.show()

# %%
# Visualize target variable
iris.target

# %%
# custer k-means
kmeans = KMeans(n_clusters=3, n_init='auto')
kmeans.fit(iris.data)
print(f'KMeans labels: {kmeans.labels_}')

# %%
# confusion matrix
result = confusion_matrix(iris.target, kmeans.labels_)
print(f'KMeans confusion matrix:\n{result}')
pltot_clusters(iris.data, kmeans.labels_, 'KMeans Clustering')

# %%
# custer dbscan
dbscan = DBSCAN(eps=0.5, min_samples=3)
dbscan_labels = dbscan.fit_predict(iris.data)
print(f'DBSCAN labels: {dbscan_labels}')

# %%
# confusion matrix
result = confusion_matrix(iris.target, dbscan_labels)
print(f'DBSCAN confusion matrix:\n{result}')
pltot_clusters(iris.data, dbscan_labels, 'DBSCAN Clustering')


# %%
# custer agglomerative clustering
agglo = AgglomerativeClustering(n_clusters=3)
agglo_labels = agglo.fit_predict(iris.data)
print(f'Agglomerative Clustering labels: {agglo_labels}')

# %%
# confusion matrix
result = confusion_matrix(iris.target, agglo_labels)
print(f'Agglomerative Clustering confusion matrix:\n{result}')
pltot_clusters(iris.data, agglo_labels, 'Agglomerative Clustering')

# %%
# dendrogram
plt.figure(figsize=(12, 6))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
linkage_matrix = linkage(iris.data, method='ward')
dendrogram(linkage_matrix,truncate_mode='lastp', p=15)
plt.axhline(y=7, color='gray',lw=1, linestyle='--')
plt.show()

# %%
