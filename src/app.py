'''

Python basic machine learning.
Copyright 2020 CHASM.ML
'''

'''

Basic Linear Regression in Python
Using Scikit learn
'''
# Importing all dependencies
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

'''

Defining a root directory.

Defining a dataset path combined with root directory

@var string
@var string
'''
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_DATASET_PATH = f'{ROOT_DIR}/assets/dataset/student_scores.csv'
CAST_DATASET_PATH = f'{ROOT_DIR}/assets/dataset/cast.csv'
ADMISSION_V0_DATASET_PATH = f'{ROOT_DIR}/assets/dataset/Admission_Predict.csv'

df = pd.read_csv(CAST_DATASET_PATH, low_memory=False)

df.fillna(method='ffill', inplace=True)

X = np.array(df['Cruise']).reshape(-1, 1)
y = np.array(df['Cruz_Sta']).reshape(-1, 1)

df.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

regr = LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='k')
plt.show()
'''

df.fillna(method='ffill', inplace=True)

# Define the data to feed the model
X = np.array(df['Hours']).reshape(-1, 1)
y = np.array(df['Scores']).reshape(-1, 1)

df.dropna(inplace=True)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the model and train the model 
regr = LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

# Plotting the results
plt.scatter(X_test, y_test, color='b')
plt.plot(X_test, y_pred, color='k')
plt.show()
'''