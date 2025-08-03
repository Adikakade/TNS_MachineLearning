import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("play_tennis.csv")  # Replace with your actual file
df.head()
df.tail()

# Create a copy of the dataset
df_encoded = df.copy()

# Encode all categorical columns
le = LabelEncoder()
for column in df_encoded.columns:
    df_encoded[column] = le.fit_transform(df_encoded[column])

print(df_encoded)

X = df_encoded.drop('PlayTennis', axis=1)
y = df_encoded['PlayTennis']

# Initialize and train model
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X, y)

# Plot the tree
plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree for Play Tennis")
plt.show()

y_pred = model.predict(X)
print("Predictions:", y_pred)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y, y_pred))
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))