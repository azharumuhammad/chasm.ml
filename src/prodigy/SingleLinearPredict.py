"""

Creating basic of linear regression
Copyright @ 2020
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

"""

Defining SingleLinearPredict classes
to create linear regression
"""


class SingleLinearPredict(object):
    def __init__(self, dataset, X, y, test_size=0.2):
        self.dataset = dataset
        self.test_size = test_size

        self.df = pd.read_csv(self.dataset)
        self.X = np.array(self.df[X]).reshape(-1, 1)
        self.y = np.array(self.df[y]).reshape(-1, 1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size
        )

        self.model = LinearRegression()

    def predict(self, v):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(v)
        
        """
        Render the value of trained model
        """
        plt.scatter(self.X_test, self.y_test)
        plt.plot(self.X_test, y_pred)
        plt.show()

