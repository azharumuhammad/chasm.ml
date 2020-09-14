'''
Python basic machine learning.
Copyright 2020 CHASM.ML
'''

'''
Basic Linear Regression in Python
Using Scikit learn
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define root directory for development purpose
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = f'{ROOT_DIR}/assets/dataset/student_scores.csv'

dataset = pd.read_csv(DATASET_PATH)

# Create train data from the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Importing scickit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

accuracy = regressor.coef_
loss = regressor.intercept_

# Printing the accuracy and the loss function
print(f'accuracy: {accuracy[0]}@loss: {loss}')

# Predict the value
y_pred = regressor.predict(X_test)

line = loss + accuracy * X_test

'''
for v in X:
    line.append(v + loss)

print(line)
'''

# Creating GUI to make readable
df = pd.DataFrame({'A': y_test, 'B': y_pred})
df.plot(x='A', y='B', style='o')
plt.plot(y_pred)
plt.title('Index')
plt.xlabel('Hours studied')
plt.ylabel('Percentage scores')
plt.show()

'''
dataset.plot(x='Hours', y='Scores', style='o')
plt.plot(y_pred)
plt.title('Index')
plt.xlabel('Hours studied')
plt.ylabel('Percentage scores')
plt.show()
'''