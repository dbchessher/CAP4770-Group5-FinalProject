# source_code/calculate_information_gain.py
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

def compute_information_gain(X, y):
    info_gain = mutual_info_classif(X, y, discrete_features=True)
    return dict(zip(X.columns, info_gain))
