import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.LinearRegression import LinearRegression

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
dataset = dataset.dropna()
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

MPG_train = train_features.pop('MPG')
train_features['MPG'] = MPG_train

MPG_test = test_features.pop('MPG')
test_features['MPG'] = MPG_test


pd.set_option('display.max_columns', None)
# for value in train_features['Displacement']:
#     print(value)


new_df = pd.concat([train_features['Horsepower'], train_features['MPG']], axis=1)
print(new_df.head())

model = LinearRegression('../params/test_single_var.yaml',new_df,new_df)
model.train()


Xplot = np.array(test_features['Horsepower']).reshape(-1, 1)
Yplot = np.array(test_features['MPG']).reshape(-1)

plt.scatter(Xplot, Yplot, color='blue', label='Data Points')

x_values = np.linspace(Xplot.min(), Xplot.max(), 100)

print(model.getM()[0])
y_values = model.getM()[0] * x_values + model.getB()
plt.scatter(x_values, y_values, color='red', label='Data Points')
plt.show()
