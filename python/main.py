import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Get the absolute path of the "src" directory
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))

# Add "src" directory to sys.path
sys.path.append(src_path)

from decision_tree import train_decision_tree
from tree_conversion import tree_to_select, tree_to_where
from tree_visualization import visualize_tree

def load_data():
    """Load a dataset from a CSV file."""
    file_path = input("Enter the path to your dataset (CSV file): ").strip()
    if not os.path.exists(file_path):
        print("Error: File not found!")
        return None

    df = pd.read_csv(file_path)
    print("\nColumns in dataset:", list(df.columns))
    return df

def main():
    """Main function to train a Decision Tree and extract rules."""
    
    # Load dataset
    df = load_data()
    if df is None:
        return

    # Select target column
    target_col = input("\nEnter the target column: ").strip()
    if target_col not in df.columns:
        print("Error: Target column not found!")
        return

    # Select feature columns
    feature_cols = input("Enter feature columns (comma-separated): ").strip().split(",")
    feature_cols = [col.strip() for col in feature_cols if col.strip() in df.columns]

    if not feature_cols:
        print("Error: No valid feature columns selected!")
        return

    # Split dataset
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Decision Tree
    max_depth = int(input("\nEnter max depth for the Decision Tree (default=3): ") or "3")
    clf = train_decision_tree(X_train, y_train, max_depth=max_depth)

    # Extract decision rules
    rules_where = tree_to_where(clf, feature_cols)
    rules_select = tree_to_select(clf, feature_cols)

    # Print rules properly
    print("\n📌 Decision Rules (WHERE conditions):")
    print(rules_where.to_string(index=False))  # Fix table format

    print("\n📌 Decision Rules (SELECT conditions):")
    print(rules_select)  # Print full expression without character breaks

    # Prompt user for save path
    save_path = input("\nEnter the file path to save the decision tree visualization (default: tree_visualization.png): ").strip()
    if not save_path:
        save_path = "tree_visualization.png"

    # Visualize the tree
    class_names = list(df[target_col].unique().astype(str))
    visualize_tree(clf, feature_cols, class_names, save_path)

    print(f"\n✅ Decision tree visualization saved as '{save_path}'.")

if __name__ == "__main__":
    main()
