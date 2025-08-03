# Logistic Regression on Breast Cancer Dataset

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Step 1: Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data        # Features (30 columns)
y = data.target      # Labels: 0 = malignant, 1 = benign

# Step 2: Split the data into training and testing sets
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Feature scaling

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Create and train the model
model = LogisticRegression(max_iter=10000)  # increase max_iter for convergence
model.fit(X_train, y_train)

# Step 4: Predict on test data
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Step 6: Print the results
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy      :", accuracy)
print("Precision     :", precision)
print("Recall        :", recall)
print("F1 Score      :", f1)

# Step 7: Predict probabilities
probabilities = model.predict_proba(X_test)
print("\nPredicted Probabilities:\n", probabilities[:5])