"""
Python basic machine learning.
Copyright 2020 CHASM.ML
"""

"""
Basic Linear Regression in Python
Using Scikit learn
"""

from src.prodigy.SingleLinearPredict import SingleLinearPredict
from src.utils import DATASET_PATH

slp = SingleLinearPredict(f"{DATASET_PATH}/student_scores.csv", "Hours", "Scores")
v = slp.X_test

slp.predict(v)