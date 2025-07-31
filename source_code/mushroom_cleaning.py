# src/models/mushroom_cleaning.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(file_path):
    columns = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
        'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
        'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
    ]
    
    df = pd.read_csv(file_path, names=columns)
    df = df.replace('?', pd.NA).dropna()
    df_encoded = df.apply(LabelEncoder().fit_transform)
    
    X = df_encoded.drop("class", axis=1)
    y = df_encoded["class"]
    
    return X, y
