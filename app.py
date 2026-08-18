
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Classifier Evaluation App", layout="wide")

st.title("📊 ML Model Evaluation Dashboard")
st.markdown("Interactive application to upload test dataset, select models, and view performance metrics.")

# Sidebar Controls
st.sidebar.header("User Controls")
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

model_choice = st.sidebar.selectbox(
    "Select Classification Model",
    ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest (Ensemble)"]
)

model_file_map = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "kNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest (Ensemble)": "models/random_forest_(ensemble).pkl"
}

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Test Dataset Preview", data.head())
    
    if "target" in data.columns:
        X_test = data.drop(columns=["target"])
        y_test = data["target"]
        
        # Load Selected Model
        try:
            model = joblib.load(model_file_map[model_choice])
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
            
            # Metric Calculation
            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, probs)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            mcc = matthews_corrcoef(y_test, preds)
            
            st.subheader(f"Metrics for {model_choice}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")
            
            st.markdown("---")
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
                
            with c2:
                st.subheader("Classification Report")
                report = classification_report(y_test, preds, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
                
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.error("Uploaded CSV must contain a 'target' column.")
else:
    st.info("Please upload a valid 'test_data.csv' to run evaluations.")
