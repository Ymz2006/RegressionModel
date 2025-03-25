import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Load and preprocess the dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
dataset = dataset.dropna()  # Drop rows with missing values
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

# Split the dataset into training and testing sets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Separate features and target variable
train_features = train_dataset.copy()
test_features = test_dataset.copy()

MPG_train = train_features.pop('MPG')
MPG_test = test_features.pop('MPG')

# Select the 'Horsepower' feature for regression
X_train = np.array(train_features['Horsepower']).reshape(-1, 1)
y_train = np.array(MPG_train).reshape(-1)
X_test = np.array(test_features['Horsepower']).reshape(-1, 1)
y_test = np.array(MPG_test).reshape(-1)

# Step 2: Normalize the data (important for gradient descent)
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

# Step 3: Implement Linear Regression from scratch
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None  # Slope (coefficient)
        self.bias = None  # Intercept

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)  # Shape: (n_features,)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            # Predictions
            y_pred = np.dot(X, self.weights) + self.bias  # Shape: (n_samples,)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))  # Shape: (n_features,)
            db = (1 / n_samples) * np.sum(y_pred - y)  # Scalar

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Step 4: Train the model
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = np.mean((y_test - y_pred) ** 2)
print(f"Mean Squared Error: {mse:.2f}")

# Step 7: Plot the results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('Linear Regression from Scratch')
plt.legend()
plt.show()