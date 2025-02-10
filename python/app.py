import streamlit as st
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

# Get the absolute path of the "src" directory
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))

# Add "src" directory to sys.path
sys.path.append(src_path)

from decision_tree import train_decision_tree
from tree_conversion import tree_to_select, tree_to_where
from tree_visualization import visualize_tree


# Streamlit App
st.set_page_config(page_title="LeafRules - Decision Tree Tool", page_icon="🌳")

st.title("Decision Tree Rule Extractor")

# Upload CSV File
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    # Select target and feature columns
    target_col = st.selectbox("Select target column", df.columns, index=len(df.columns)-1)
    feature_cols = st.multiselect("Select feature columns", df.columns, default=[col for col in df.columns if col != target_col])

    if target_col and feature_cols:
        # Train/test split
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Decision Tree training
        max_depth = st.slider("Select max depth of Decision Tree", min_value=1, max_value=10, value=3)
        clf = train_decision_tree(X_train, y_train, max_depth=max_depth)

        # Extract decision rules
        rules_where = tree_to_where(clf, feature_cols)
        rules_select = tree_to_select(clf, feature_cols)

        # Display rules
        st.subheader("📌 Decision Rules (WHERE conditions)")
        st.dataframe(rules_where)

        st.subheader("📌 Decision Rules (SELECT conditions)")
        st.code(rules_select, language="python")

        # Save and visualize tree
        tree_image_path = "tree_visualization.png"
        class_names = list(df[target_col].unique().astype(str))
        visualize_tree(clf, feature_cols, class_names, tree_image_path)

        # Show tree image
        if os.path.exists(tree_image_path):
            st.image(tree_image_path, caption="Decision Tree Visualization", use_container_width=True)
