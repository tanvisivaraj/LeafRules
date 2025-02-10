import pandas as pd

def tree_to_select(tree, feature_names):
    """
    Convert a trained Decision Tree into NumPy select conditions.

    Parameters:
    - tree (DecisionTreeClassifier): Trained Decision Tree model
    - feature_names (list): List of feature names

    Returns:
    - str: NumPy select expression
    """
    tree_ = tree.tree_
    conditions = []
    values = []

    def recurse(node, condition):
        if tree_.feature[node] != -2:  # Not a leaf node
            feature = feature_names[tree_.feature[node]]
            threshold = repr(tree_.threshold[node])  # Prevents floating-point errors

            # Left child (feature <= threshold)
            recurse(tree_.children_left[node], condition + [f"({feature} <= {threshold})"])

            # Right child (feature > threshold)
            recurse(tree_.children_right[node], condition + [f"({feature} > {threshold})"])
        else:  # Leaf node
            conditions.append(" & ".join(condition))
            values.append(str(tree_.value[node].argmax()))  # Get class index

    recurse(0, [])

    # Formatting np.select output for readability
    conditions_str = ",\n    ".join(f"[{cond}]" for cond in conditions)
    values_str = ",\n    ".join(values)

    return f"np.select([\n    {conditions_str}\n], [\n    {values_str}\n], default=-1)"


def tree_to_where(tree, feature_names):
    """
    Converts a trained Decision Tree into readable 'WHERE' conditions and prints it as a table.
    """
    rules = []

    # Access the trained decision tree structure
    tree_ = tree.tree_

    def recurse(node, conditions=[]):
        """Recursively traverse the tree and collect rules."""
        if tree_.children_left[node] == -1:  # Leaf node
            prediction = tree_.value[node].argmax()  # Get predicted class
            rules.append({
                "Condition": " & ".join(conditions),
                "Prediction": prediction
            })
            return

        # Get feature and threshold
        feature = feature_names[tree_.feature[node]]
        threshold = tree_.threshold[node]

        # Traverse left (≤ threshold)
        recurse(tree_.children_left[node], conditions + [f"{feature} ≤ {threshold:.2f}"])

        # Traverse right (> threshold)
        recurse(tree_.children_right[node], conditions + [f"{feature} > {threshold:.2f}"])

    recurse(0)

    # Convert to a Pandas DataFrame for better readability
    df_rules = pd.DataFrame(rules)

    # Print as a table in Jupyter Notebook
    from IPython.display import display
    display(df_rules)

    return df_rules
