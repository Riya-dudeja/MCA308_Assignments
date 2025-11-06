# Naive Bayes Classifier using GaussianNB (Iris Dataset)

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Creating the Gaussian Naive Bayes model
nb_model = GaussianNB()

# Training the model
nb_model.fit(X_train, y_train)

# Predicting on the test data
y_pred = nb_model.predict(X_test)

# Evaluating model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.3f}\n")

print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion matrix for better understanding
plt.figure(figsize=(5,4))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    cmap="BuGn",
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.title("Confusion Matrix (Naive Bayes Classifier)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
