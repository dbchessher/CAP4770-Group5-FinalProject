import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up paths
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "../data/mushrooms.csv")
visuals_dir = os.path.join(base_dir, "../visuals")

# Create visuals directory if it doesn't exist
os.makedirs(visuals_dir, exist_ok=True)

# Load the dataset
df = pd.read_csv(csv_path)

# Class distribution
print("\nClass distribution (e = edible, p = poisonous):")
print(df['class'].value_counts())

# === Save Plot 1: Class Distribution ===
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='class', hue='class', palette='Set2', legend=False)
plt.title('Distribution of Edible vs. Poisonous Mushrooms')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, "class_distribution.png"))
plt.close()

# === Save Plot 2–5: Top Feature Countplots ===
top_features = ['odor', 'gill-color', 'spore-print-color', 'habitat']

for feature in top_features:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=feature, hue='class', palette='Set1')
    plt.title(f'{feature.capitalize()} by Mushroom Class')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = f"{feature.replace('-', '_')}_by_class.png"
    plt.savefig(os.path.join(visuals_dir, filename))
    plt.close()
