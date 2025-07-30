# ====== Standard Library Imports ====================================================================================
import os
from typing import List, Dict, Union
# ====== External Library Imports ====================================================================================
import pandas as pd
import numpy as np
from pandas.core.interchange import dataframe
from pandas.core.interchange.dataframe_protocol import DataFrame

# ====== Internal Library Imports ====================================================================================
from dictionaries import data_column_maps

# ====== Constants ===================================================================================================
# File paths
base_dir = os.path.dirname("")
csv_path = os.path.join(base_dir, "../data/mushrooms.csv")
visuals_dir = os.path.join(base_dir, "../visuals")


# ====== Helper Functions ============================================================================================

# Load data
def load_data(filename: str) -> pd.DataFrame:
    """
    Load mushroom data from csv file

    Parameters
    ----------
    filename

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(csv_path, na_values="None")
    return df


# Map data using dictionary values
def map_dictionaries(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Map mushroom data columns into dictionary columns

    Parameters
    ----------
    dataframe

    Returns
    -------
    dataframe
    """

    dataframe.columns = list(data_column_maps.keys())
    for col, mapping in data_column_maps.items():
        dataframe[col] = dataframe[col].map(mapping)
    # dataframe['edibility_num'] = dataframe['edibility'].map({'edible': 1, 'poisonous': 0})
    return dataframe


# ====== ID3 Functions ================================================================================================
def is_homogenous(data: pd.DataFrame, label_index: int = -1) -> bool:
    """Check if all labels in the dataset are identical, i.e. Entropy = 0"""
    return data.iloc[:, label_index].nunique() == 1


def majority_label(data: Union[pd.DataFrame, pd.Series], label_index: int = -1) -> str:
    """Find the most common label in the dataset"""
    if isinstance(data, pd.Series):
        return str(data.mode()[0])
    return str(data.iloc[:, label_index].mode()[0])


def entropy(data: pd.DataFrame, class_index: int = -1) -> float:
    """Calculate entropy of a dataset"""
    value_counts = data.iloc[:, class_index].value_counts()
    proportions = value_counts / len(data)
    return -np.sum(proportions * np.log2(proportions))


def information_gain(data: pd.DataFrame, attribute_index: int, class_index: int = -1) -> float:
    """Calculate information gain for a given attribute"""
    total_entropy = entropy(data, class_index)

    weighted_entropy = 0
    for value in data.iloc[:, attribute_index].unique():
        subset = data[data.iloc[:, attribute_index] == value]
        weight = len(subset) / len(data)
        weighted_entropy += weight * entropy(subset, class_index)

    return total_entropy - weighted_entropy


def pick_best_attribute(data: pd.DataFrame, attributes: List[str] = None) -> str:
    """Find the attribute with the highest information gain"""
    if attributes is None:
        attributes = list(data.columns[:-1])

    best_gain = -1
    best_attr = None

    for attr in attributes:
        attr_index = data.columns.get_loc(attr)
        gain = information_gain(data, attr_index)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr

    return best_attr


def clear_redundant_children(node: Dict) -> Union[Dict, str]:
    """Remove redundant children if all are the same"""
    if isinstance(node, str):
        return node

    children_values = list(node["children"].values())
    if all(val == children_values[0] for val in children_values):
        return children_values[0]
    return node


def id3(data: pd.DataFrame, attributes: List[str], default: str = None, label_col: str = None) -> Union[Dict, str]:
    """Build a decision tree using the ID3 algorithm"""
    if label_col is None:
        label_col = data.columns[-1]

    if len(data) == 0:
        return default

    if is_homogenous(data, data.columns.get_loc(label_col)):
        return str(data[label_col].iloc[0])

    if len(attributes) == 0:
        return majority_label(data, data.columns.get_loc(label_col))

    best_attr = pick_best_attribute(data[attributes + [label_col]])
    node = {"root": best_attr, "children": {}}
    default_label = majority_label(data, data.columns.get_loc(label_col))

    for value in data[best_attr].unique():
        subset = data[data[best_attr] == value]
        next_attrs = [attr for attr in attributes if attr != best_attr]
        child = id3(subset, next_attrs, default_label, label_col)
        node["children"][str(value)] = child

    return clear_redundant_children(node)


def predict(decision_tree: Dict, data_row: pd.Series) -> str:
    """Predict the class for a single data instance"""
    if isinstance(decision_tree, str):
        return decision_tree

    root_attr = decision_tree["root"]
    value = str(data_row[root_attr])

    if value in decision_tree["children"]:
        next_node = decision_tree["children"][value]
    else:
        # Default to majority class if value not seen
        leaves = flatten(decision_tree)
        return majority_label(pd.Series(leaves))

    if isinstance(next_node, str):
        return next_node
    else:
        return predict(next_node, data_row)


def classify(decision_tree: Union[Dict, str], test_data: pd.DataFrame) -> List[str]:
    """Classify all instances in the test dataset"""
    predictions = []
    for _, row in test_data.iterrows():
        if isinstance(decision_tree, str):
            predictions.append(decision_tree)
        else:
            predictions.append(predict(decision_tree, row))
    return predictions


def classification_error(decision_tree: Dict, test_data: pd.DataFrame, label_col: str = None) -> float:
    """Calculate classification error rate"""
    if label_col is None:
        label_col = test_data.columns[-1]

    true_labels = test_data[label_col].tolist()
    predictions = classify(decision_tree, test_data.drop(columns=[label_col]))

    return evaluate(true_labels, predictions)


def flatten(decision_tree: Union[Dict, str]) -> List[str]:
    """Flatten the tree to get all leaf values"""
    if isinstance(decision_tree, str):
        return [decision_tree]

    leaves = []
    for key, value in decision_tree.items():
        if key == "root":
            continue
        elif isinstance(value, dict):
            leaves.extend(flatten(value))
        else:
            leaves.append(value)
    return leaves


def evaluate(y_true: List, y_pred: List) -> float:
    """Calculate error rate"""
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return 1 - (correct / len(y_true))


def print_tree(decision_tree: Dict, prefix: str = '', spacing: int = 4):
    """Pretty print the decision tree"""
    clr = {"pnk": "\033[95m", "ylo": "\033[93m", "grn": "\033[92m", "end": "\033[0m"}

    if isinstance(decision_tree, str):
        print(f"{prefix}└──>{clr['grn']}{decision_tree}{clr['end']}")
        return

    root = decision_tree["root"]
    if prefix == "":
        print(f"-({clr['pnk']}{root}{clr['end']})")
        prefix = ' ' * spacing
    else:
        print(f"{prefix}└({clr['pnk']}{root}{clr['end']})")
        prefix = prefix + ' ' * spacing * 2

    children = decision_tree["children"]
    child_items = list(children.items())

    for i, (value, child) in enumerate(child_items):
        is_last = (i == len(child_items) - 1)
        connector = "└" if is_last else "├"

        if isinstance(child, str):
            print(f"{prefix}{connector}┬[{clr['ylo']}{value}{clr['end']}]?")
            print(f"{prefix}{'  ' if is_last else '| '}└──>{clr['grn']}{child}{clr['end']}")
        else:
            print(f"{prefix}{connector}-[{clr['ylo']}{value}{clr['end']}]?")
            new_prefix = prefix + ("  " if is_last else "| ")
            print_tree(child, new_prefix)


data = load_data(csv_path)
data_named = map_dictionaries(data)
attributes = [k for k in data_named if k != "edibility"]
tree = id3(data_named, attributes, label_col='edibility')
print_tree(tree)

"""
-(odor)
    ├┬[almond]?
    | └──>edible
    ├┬[anise]?
    | └──>edible
    ├┬[pungent]?
    | └──>poisonous
    ├-[none]?
    | └(spore_print_color)
    |         ├┬[brown]?
    |         | └──>edible
    |         ├┬[black]?
    |         | └──>edible
    |         ├-[white]?
    |         | └(stalk_root)
    |         |         ├┬[nan]?
    |         |         | └──>edible
    |         |         ├-[bulbous]?
    |         |         | └(cap_color)
    |         |         |         ├┬[cinnamon]?
    |         |         |         | └──>edible
    |         |         |         ├┬[brown]?
    |         |         |         | └──>edible
    |         |         |         ├┬[white]?
    |         |         |         | └──>poisonous
    |         |         |         ├┬[pink]?
    |         |         |         | └──>edible
    |         |         |         └┬[gray]?
    |         |         |           └──>edible
    |         |         └┬[club]?
    |         |           └──>poisonous
    |         ├┬[chocolate]?
    |         | └──>edible
    |         ├┬[green]?
    |         | └──>poisonous
    |         ├┬[orange]?
    |         | └──>edible
    |         ├┬[yellow]?
    |         | └──>edible
    |         └┬[buff]?
    |           └──>edible
    ├┬[foul]?
    | └──>poisonous
    ├┬[creosote]?
    | └──>poisonous
    ├┬[fishy]?
    | └──>poisonous
    ├┬[spicy]?
    | └──>poisonous
    └┬[musty]?
      └──>poisonous
"""