# source_code/random_forest.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def run_random_forest(X, y, output_path="visuals/random_forest_importance.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=X.columns)
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return model, X_test, y_test
