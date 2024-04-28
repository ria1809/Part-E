import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc

def remove_outlier(ds, col):
    quart1 = ds[col].quantile(0.25)
    quart3 = ds[col].quantile(0.75)
    IQR = quart3 - quart1
    low_val = quart1 - 1.5 * IQR
    high_val = quart3 + 1.5 * IQR
    df_out = ds.loc[(ds[col] > low_val) & (ds[col] < high_val)]
    return df_out


filepath = r'C:\Users\riaro\Downloads\glass+identification\glass.data'

data = pd.read_csv(filepath, delimiter=',')

# Remove outliers
outlier_removed_data = remove_outlier(data, "Ca")
clustering_dataset = outlier_removed_data.drop(['Type'], axis=1)

# Creating a dendrogram
plt.figure(figsize=(8, 6))
dendrogram(linkage(clustering_dataset, method='ward'))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.axhline(y=600, color='r', linestyle='--', label='Threshold')
plt.legend() 
plt.show()  

# Scatter plot of Mg and Ca without clustering algorithm
plt.figure(figsize=(8, 6))
plt.scatter(clustering_dataset['Mg'], clustering_dataset['Ca'], color='blue')
plt.xlabel('Mg')
plt.ylabel('Ca')
plt.title('Scatter Plot of Mg and Ca')
plt.show()

# Scale the features
scaler = StandardScaler()
clustering_dataset_scaled = scaler.fit_transform(clustering_dataset)
linkage_criteria = ['ward', 'complete', 'average']


for criterion in linkage_criteria:
    hierarchical_clustering = AgglomerativeClustering(n_clusters=2, linkage=criterion)
    labels = hierarchical_clustering.fit_predict(clustering_dataset_scaled)

   
    plt.figure(figsize=(8, 6))
    plt.title(f'Agglomerative Clustering with {criterion} linkage')
    plt.xlabel('Mg')
    plt.ylabel('Ca')
    plt.scatter(clustering_dataset_scaled[:, 3], clustering_dataset_scaled[:, 7], c=labels, cmap='rainbow')
    plt.show()


# Optimal number of clusters is 2
aggl_clust2 = AgglomerativeClustering(n_clusters=2, linkage='ward')

plt.figure(figsize=(8, 6))
plt.scatter(clustering_dataset_scaled[:,3], clustering_dataset_scaled[:,7], 
           c=aggl_clust2.fit_predict(clustering_dataset_scaled), cmap='rainbow')
plt.xlabel("Mg")
plt.ylabel("Ca")
plt.title('Agglomerative Clustering (n_clusters=2)')
plt.show()
