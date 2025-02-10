import matplotlib.pyplot as plt
from sklearn import tree

def visualize_tree(clf, feature_names, class_names, save_path):
    """
    Plot and save the visualization of a trained Decision Tree.

    Parameters:
    - clf (DecisionTreeClassifier): Trained Decision Tree model
    - feature_names (list): List of feature names
    - class_names (list): List of class names
    - save_path (str): Path to save the tree visualization

    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    
    tree.plot_tree(
        clf, 
        feature_names=feature_names, 
        class_names=class_names, 
        filled=True, 
        rounded=True, 
        fontsize=10
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Tree visualization saved to {save_path}")
    
    plt.show()