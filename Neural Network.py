import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


filename = r'C:\Users\riaro\Downloads\glass+identification\glass.data'
data = pd.read_csv(filename, delimiter=',')

# Encode target labels
my_class_labels_coded = preprocessing.LabelEncoder()
my_class_labels_coded.fit(data['Type'])


my_working_dataset_8_features = data.drop(['Type'], axis=1)
targets = data['Type']


training_proportion = 0.7
X_train, X_test, y_train, y_test = train_test_split(my_working_dataset_8_features, targets, test_size=1.0 - training_proportion, random_state=42)


num_train_samples = X_train.shape[0]


num_test_samples = X_test.shape[0]

print("Number of samples in the training set:", num_train_samples)
print("Number of samples in the test set:", num_test_samples)

# occurrences training set
train_class_counts = {}
for label in set(y_train):
    train_class_counts[label] = sum(y_train == label)

# occurrences in test set
test_class_counts = {}
for label in set(y_test):
    test_class_counts[label] = sum(y_test == label)


print("Class counts in training set:")
for label, count in train_class_counts.items():
    print(f"Class {label}: {count} samples")


print("\nClass counts in test set:")
for label, count in test_class_counts.items():
    print(f"Class {label}: {count} samples")

# Define hyperparameters for MLPClassifier
mlp_params = [
    {'hidden_layer_sizes': (50,), 'max_iter': 1000, 'activation': 'relu', 'solver': 'adam'},
    {'hidden_layer_sizes': (100,50), 'max_iter': 800, 'activation': 'relu', 'solver': 'adam'},
    {'hidden_layer_sizes': (100,100), 'max_iter': 500, 'activation': 'relu', 'solver': 'adam'}
]

# Perform three experiments with different hyperparameters
for i, params in enumerate(mlp_params, 1):
    print(f"Experiment {i} with hyperparameters: {params}")
    
    
    model = MLPClassifier(**params, verbose=0)
    model.fit(X_train, y_train)

    
    y_predictions = model.predict(X_test)

    
    print(classification_report(y_test, y_predictions, zero_division=0))

    # confusion matrix
    my_confusion_matrix = confusion_matrix(y_test, y_predictions)

  
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(my_confusion_matrix, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted class labels')
    ax.set_ylabel('True class labels')
    ax.set_title('Confusion matrix')
    ax.xaxis.set_ticklabels(my_class_labels_coded.classes_)
    ax.yaxis.set_ticklabels(my_class_labels_coded.classes_)
    plt.show()
