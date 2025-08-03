# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('car_data.csv')

# Display first few rows
print(df.head())
print(df.describe())
print(df.info())

# Data cleaning and preprocessing
# Check for missing values
print("\n Checking for missing values:\n")
print(df.isnull().sum())
df.dropna(inplace=True)

# Convert 'year' to vehicle age
df['car_age'] = 2025 - df['year']


# Encoding categorical variables
le_fuel = LabelEncoder()
le_seller = LabelEncoder()
le_transmission = LabelEncoder()


df['fuel'] = le_fuel.fit_transform(df['fuel'])
print(df['fuel'])
df['seller_type'] = le_seller.fit_transform(df['seller_type'])
df['transmission'] = le_transmission.fit_transform(df['transmission'])

# Selecting relevant features
X = df[['car_age', 'km_driven', 'fuel', 'seller_type', 'transmission']]
y = df['selling_price']

# Splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plotting Actual vs Predicted
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid()
plt.show()