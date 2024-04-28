import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report


filepath = r'C:\Users\riaro\Downloads\glass+identification\glass.data'
data = pd.read_csv(filepath, delimiter=',')


X = data.drop(['Type'], axis=1)
y = data['Type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_train_samples = X_train.shape[0]


num_test_samples = X_test.shape[0]

print("Number of samples in the training set:", num_train_samples)
print("Number of samples in the test set:", num_test_samples)

# occurrences training set
train_class_counts = {}
for label in set(y_train):
    train_class_counts[label] = sum(y_train == label)

# occurrences  in test set
test_class_counts = {}
for label in set(y_test):
    test_class_counts[label] = sum(y_test == label)


print("Class counts in training set:")
for label, count in train_class_counts.items():
    print(f"Class {label}: {count} samples")


print("\nClass counts in test set:")
for label, count in test_class_counts.items():
    print(f"Class {label}: {count} samples")

# Experiments 
experiments = [
    {"name": "Experiment 1", "k": 3, "metric": 'euclidean'},
    {"name": "Experiment 2", "k": 3, "metric": 'manhattan'},
    {"name": "Experiment 3", "k": 3, "metric": 'minkowski'}
]

# Run experiments
for exp in experiments:
    knn_classifier = KNeighborsClassifier(n_neighbors=exp["k"], metric=exp["metric"], **exp.get("extra_params", {}))
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    
    print(f"{exp['name']} - Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n")
    # Plot confusion matrix as heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, len(y.unique())+1), yticklabels=range(1, len(y.unique())+1))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"{exp['name']} - Confusion Matrix")
    plt.show()
