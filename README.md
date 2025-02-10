# Decision Tree Rule Extractor

## 📌 Overview
This is a **Streamlit-based web application** that allows users to:
- Upload a CSV dataset
- Select a target column and feature columns
- Train a Decision Tree classifier
- Extract SQL-like decision rules (`WHERE` and `SELECT` conditions)
- Visualize the trained Decision Tree

## 🚀 Features
- **Easy CSV Upload**: Users can upload their dataset via Streamlit's file uploader.
- **Automatic Target Selection**: The last column of the dataset is pre-selected as the target column.
- **Interactive Model Training**: Users can choose feature columns and adjust the tree depth using a slider.
- **Decision Rule Extraction**: Converts the trained Decision Tree into SQL-like conditions.
- **Tree Visualization**: Displays the Decision Tree structure as an image.

## 🛠 Installation & Setup
### **1. Clone the Repository**
```sh
git clone https://github.com/tanvisivaraj/LeafRules.git
cd LeafRules
```

### 2. Create a Virtual Environment
Create a virtual environment named `leafrulesenv`:
```bash
python -m venv leafrulesenv
```

### 3. Activate the Virtual Environment
- **Windows:**
  ```bash
  leafrulesenv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source leafrulesenv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🎯 Usage
### **Run the Streamlit App**
```sh
streamlit run app.py
```
This will open the web app in your browser.

### **How to Use the App?**
1. Upload a CSV dataset.
2. Select the **target column** (pre-selected as the last column by default).
3. Select **feature columns** for training.
4. Adjust the **max depth** of the decision tree using the slider.
5. View extracted SQL-like rules.
6. Visualize the Decision Tree structure.

## 📚 Dependencies
- `streamlit`
- `pandas`
- `scikit-learn`
- `graphviz`

## 💡 Future Improvements
- Add support for **multi-class classification**.
- Allow users to download extracted SQL rules.
- Enhance visualization with interactive tree structures.

## 🤝 Contributing
Pull requests are welcome! Feel free to open an issue for discussion.
