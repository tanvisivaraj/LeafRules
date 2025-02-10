import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X, y, max_depth=3, random_state=42, save_path=None):
    """
    Train a Decision Tree classifier.

    Parameters:
    - X (array-like): Features
    - y (array-like): Target variable
    - max_depth (int): Maximum depth of the tree
    - random_state (int): Random seed for reproducibility
    - save_path (str): Path to save the trained model (optional)

    Returns:
    - clf (DecisionTreeClassifier): Trained model
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)

    if save_path:
        joblib.dump(clf, save_path)
        print(f"Model saved to {save_path}")

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    
    print(f"Train Accuracy: {train_acc:.2f}")
    print(f"Test Accuracy: {test_acc:.2f}")

    return clf