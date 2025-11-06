# Rule-Based Classification using OneR Algorithm (Iris Dataset)

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=42, stratify=y
)

def one_r(X, y, bins=5):
    rules = {}
    best_feature = None
    best_error = float('inf')

    for feature in X.columns:
        # Convert continuous values into bins
        X_binned = pd.cut(X[feature], bins=bins, labels=False)
        feature_rule = {}

        # For each bin, find the most common class
        for val in X_binned.unique():
            most_common_class = y[X_binned == val].mode()[0]
            feature_rule[val] = most_common_class

        # Predict on training data to calculate error
        y_pred = X_binned.map(feature_rule)
        error = sum(y_pred != y)

        # Keep track of best performing feature
        rules[feature] = feature_rule
        if error < best_error:
            best_error = error
            best_feature = feature

    return best_feature, rules[best_feature], best_error / len(y)

# Train the OneR model
best_feature, rule, error_rate = one_r(X_train, y_train)
print("Best Feature Selected:", best_feature)
print("Rule (Bin → Class):")
for k, v in rule.items():
    print(f"  Bin {k}: Class {v}")
print(f"\nTraining Error Rate: {error_rate:.3f}")

# Prediction Function
def predict_one_r(X, feature, rule, bins=5, default_class=None):
    X_binned = pd.cut(X[feature], bins=bins, labels=False)
    return X_binned.map(rule).fillna(default_class).astype(int)

# Set default class as most frequent one
default_class = y_train.mode()[0]

# Predict on test data
y_pred = predict_one_r(X_test, best_feature, rule, default_class=default_class)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.3f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix visualization
plt.figure(figsize=(5,4))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    cmap="YlGnBu",
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.title("Confusion Matrix (OneR Classifier)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()