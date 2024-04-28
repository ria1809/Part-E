import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def remove_outlier(ds, col):
    quart1 = ds[col].quantile(0.25)
    quart3 = ds[col].quantile(0.75)
    IQR = quart3 - quart1 #Interquartile
    range
    low_val = quart1 - 1.5*IQR
    high_val = quart3 + 1.5*IQR
    df_out = ds.loc[(ds[col] > low_val) & (ds[col] < high_val)]
    return df_out


filepath = r'C:\Users\riaro\Downloads\glass+identification\glass.data'

data = pd.read_csv(filepath, delimiter=',')

counts = data['Type'].value_counts()

#bar chart for class distribution
counts.plot(kind='bar')
plt.title('Data objects in each class')
plt.xlabel('Class number')
plt.ylabel('number of data objects')
plt.show()

##shows null values/missing values. There were none
data.info()

## shows duplicates. There were none
data[(data.duplicated())]

##used to find basic statistical values
print(data.describe() )

## prints data for the first 4 features 
X = data.iloc[:, :10]
print(X)

#Boxplot for feature 7. Calcium
plt.boxplot(X.iloc[:, 7], boxprops=dict(color='blue'))  
plt.show()

#outliers on boxplot for all features
outliers=plt.boxplot(X.iloc[:,7])["fliers"][0].get_data()[1]
print(outliers)

outlier_removed_data=remove_outlier(data,"Ca")

plt.boxplot(outlier_removed_data['Ca'], boxprops=dict(color='red'))
plt.title('Boxplot of Ca without Outliers')
plt.show()

#Histogram for skewed data. exploring data within features
x = data.iloc[:, :4]
column = data['Na']
plt.xlabel('Na')
plt.hist(column)
plt.show()

##scatterplots between two variables 
plt.xlabel('Mg')
plt.ylabel('RI')
plt.scatter(data['RI'], data['Mg'], color='blue', alpha=0.5)
plt.show()

#Heatmap 
updated_data = outlier_removed_data.drop(['ID_number','Type'], axis=1) # drops ID_number and Type

correlation_matrix = updated_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Heatmap of Oxide Correlations')
plt.show()





