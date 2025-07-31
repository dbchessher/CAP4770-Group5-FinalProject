# source_code/super_tree.py
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from supertree import SuperTree

def run_supertree(X, y, output_path="visuals/super_tree.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        ccp_alpha=0.0,
        random_state=42
    )
    model.fit(X_train, y_train)

    plt.figure(figsize=(30, 15))
    plot_tree(model, filled=True, feature_names=X.columns, class_names=["Edible", "Poisonous"])
    plt.title("SuperTree: Full Depth Decision Tree")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return model, X_test, y_test