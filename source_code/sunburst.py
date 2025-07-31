import os
import pandas as pd
import plotly.express as px

def generate_sunburst(data_path="data/agaricus-lepiota.data", output_path="visuals/sunburst_chart.html"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    columns = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
        'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
        'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
    ]

    df = pd.read_csv(data_path, names=columns)
    df = df[df['odor'] != '?']
    df['class_label'] = df['class'].map({'e': 'Edible', 'p': 'Poisonous'})

    # Group data by key categories
    grouped = df.groupby(['odor', 'gill-spacing', 'spore-print-color', 'class_label']).size().reset_index(name='count')

    fig = px.sunburst(
        grouped,
        path=['odor', 'gill-spacing', 'spore-print-color'],
        values='count',
        color='class_label',
        color_discrete_map={'Edible': 'green', 'Poisonous': 'red'},
        title="Sunburst Chart: Odor → Gill-Spacing → Spore-Print Color"
    )

    fig.write_html(output_path)
    print(f"✅ Sunburst chart saved to {output_path}")
