from src.LinearRegression import LinearRegression
import numpy as np
import pandas as pd

num_pairs = 1000
a = np.random.randint(1, 101, size=num_pairs)  # First number
b = np.random.randint(1, 101, size=num_pairs)  # Second number
sum_ab = a + b  # Sum of the pairs


# Create a Pandas DataFrame
df = pd.DataFrame({
    'a': a,
    'b': b,
    'sum': sum_ab
})

inputs = df.iloc[ :700, :]
outputs = df.iloc[ 700: , :]

model = LinearRegression('../params/test_model_parameters.yaml',inputs,outputs)
model.train()
model.evaluate_model()