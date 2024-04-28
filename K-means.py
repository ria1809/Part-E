import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics

def remove_outlier(ds, col):
    quart1 = ds[col].quantile(0.25)
    quart3 = ds[col].quantile(0.75)
    IQR = quart3 - quart1 #Interquartile range
    low_val = quart1 - 1.5*IQR
    high_val = quart3 + 1.5*IQR
    df_out = ds.loc[(ds[col] > low_val) & (ds[col] < high_val)]
    return df_out


filepath = r'C:\Users\riaro\Downloads\glass+identification\glass.data'
data = pd.read_csv(filepath, delimiter=',')

outlier_removed_data=remove_outlier(data,"Ca")
clustering_dataset = outlier_removed_data.drop(['Type'], axis=1)

#Creating a list of silhouette coefficient values
Silhouette_coefficient = []

for i in range(2,7):
   kmeans_model = KMeans(n_clusters = i, init = "k-means++")
   kmeans_model.fit(clustering_dataset)

   Silhouette_coefficient.append(metrics.silhouette_score(clustering_dataset, kmeans_model.labels_))
print(Silhouette_coefficient)


#Visualisation
plt.figure(figsize=(4,4))
plt.plot(range(2,7), Silhouette_coefficient)
plt.xlabel('Cluster number')
plt.ylabel('Silhouette coefficient')
plt.show()

#optimal number of clusters is k = 2 so let's have a look at the results 
kmeans_model = KMeans(n_clusters=2, init = "k-means++").fit(clustering_dataset)

#acquiring cluster centers
centroids = kmeans_model.cluster_centers_

#make predictions for cluster membership of data objects
y_pred = kmeans_model.predict(clustering_dataset)
plt.subplot(2,2,1)                            
plt.scatter(clustering_dataset['Mg'], clustering_dataset['Ca'], s = 10, c = y_pred)
plt.scatter(centroids[0,0], centroids[0,1], s = 100, c = 'g', marker = 's')
plt.scatter(centroids[1,0], centroids[1,1], s = 100, c = 'g', marker = 's')
plt.subplot(2,2,2)                           
plt.scatter(clustering_dataset['Mg'], clustering_dataset['RI'], s = 10, c = y_pred)
plt.scatter(centroids[0,0], centroids[0,1], s = 100, c = 'g', marker = 's')
plt.scatter(centroids[1,0], centroids[1,1], s = 100, c = 'g', marker = 's')
plt.subplot(2,2,3)                            
plt.scatter(clustering_dataset['Ca'], clustering_dataset['RI'], s = 10, c = y_pred)
plt.scatter(centroids[0,0], centroids[0,1], s = 100, c = 'g', marker = 's')
plt.scatter(centroids[1,0], centroids[1,1], s = 100, c = 'g', marker = 's')
plt.subplot(2,2,4)                            
plt.scatter(clustering_dataset['Mg'], clustering_dataset['Na'], s = 10, c = y_pred)
plt.scatter(centroids[0,0], centroids[0,1], s = 100, c = 'g', marker = 's')
plt.scatter(centroids[1,0], centroids[1,1], s = 100, c = 'g', marker = 's')
plt.show()

