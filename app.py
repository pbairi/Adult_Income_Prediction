"""
Adult Income Classification - Streamlit App
Predicts whether an individual's annual income exceeds $50K
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================
# Page Configuration
# =========================================

st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">💼 Adult Income Prediction - ML Model Comparison</p>', unsafe_allow_html=True)
st.markdown("---")

# =========================================
# Sidebar
# =========================================

with st.sidebar:
    st.markdown("## 📊 Navigation")
    page = st.radio("Select Page:", ["📈 Model Overview", "🔍 Model Prediction", "📉 Model Comparison"])
    
    st.markdown("---")
    st.markdown("### 📌 About")
    st.info("""
    **Machine Learning Assignment 2**
    
    This app demonstrates 6 classification models on the Adult Census Income dataset.
    
    **Models:**
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors
    - Naive Bayes
    - Random Forest
    - XGBoost
    """)

# =========================================
# Load Functions
# =========================================

@st.cache_data
def load_results():
    try:
        return pd.read_csv('model_results.csv')
    except:
        st.error("Error: model_results.csv not found!")
        return None

@st.cache_resource
def load_model(model_name):
    try:
        filename = f"{model_name.replace(' ', '_').lower()}_model.pkl"
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler():
    try:
        with open('scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

@st.cache_data
def load_test_data():
    try:
        X_test = pd.read_csv('test_data.csv')
        y_test = pd.read_csv('test_labels.csv')
        return X_test, y_test
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return None, None

# Load data
results_df = load_results()
scaler = load_scaler()
X_test_full, y_test_full = load_test_data()

# =========================================
# Page 1: Model Overview
# =========================================

if page == "📈 Model Overview":
    st.markdown('<p class="sub-header">📊 Model Performance Metrics</p>', unsafe_allow_html=True)
    
    if results_df is not None:
        # Display metrics table
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Best model highlight
        best_model_idx = results_df['Accuracy'].idxmax()
        best_model = results_df.loc[best_model_idx, 'Model']
        best_accuracy = results_df.loc[best_model_idx, 'Accuracy']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Best Model", best_model)
        with col2:
            st.metric("📈 Best Accuracy", f"{best_accuracy:.4f}")
        with col3:
            avg_accuracy = results_df['Accuracy'].mean()
            st.metric("📊 Average Accuracy", f"{avg_accuracy:.4f}")
        
        st.markdown("---")
        
        # Visualizations
        st.markdown('<p class="sub-header">📉 Performance Visualizations</p>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 Metric Comparison", "📈 Radar Chart", "🎯 Heatmap"])
        
        with tab1:
            # Bar chart for all metrics
            metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
            
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=metrics_to_plot
            )
            
            positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
            
            for idx, metric in enumerate(metrics_to_plot):
                row, col = positions[idx]
                fig.add_trace(
                    go.Bar(x=results_df['Model'], y=results_df[metric], name=metric,
                          marker_color='lightblue'),
                    row=row, col=col
                )
            
            fig.update_layout(height=800, showlegend=False, title_text="Model Performance Across All Metrics")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Radar chart
            fig = go.Figure()
            
            for idx, row in results_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row['Accuracy'], row['AUC'], row['Precision'], 
                       row['Recall'], row['F1'], row['MCC']],
                    theta=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                    fill='toself',
                    name=row['Model']
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Model Performance Radar Chart"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Heatmap
            metrics_df = results_df.set_index('Model')
            fig = px.imshow(metrics_df.T, 
                           labels=dict(x="Model", y="Metric", color="Score"),
                           color_continuous_scale="Blues",
                           aspect="auto")
            fig.update_layout(title="Performance Heatmap")
            st.plotly_chart(fig, use_container_width=True)

# =========================================
# Page 2: Model Prediction
# =========================================

elif page == "🔍 Model Prediction":
    st.markdown('<p class="sub-header">🔮 Make Predictions with Trained Models</p>', unsafe_allow_html=True)
    
    if results_df is not None:
        # Model selection
        model_name = st.selectbox("Select Model:", results_df['Model'].tolist())
        
        # File upload
        st.markdown("### 📤 Upload Test Data")
        uploaded_file = st.file_uploader("Upload CSV file (with or without 'income' column)", type=['csv'])
        
        if uploaded_file is not None:
            # Load uploaded data
            test_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Data loaded successfully! Shape: {test_data.shape}")
            
            # Show data preview
            with st.expander("📋 View Data Preview"):
                st.dataframe(test_data.head(10))
            
            # Check if income column exists
            has_labels = 'income' in test_data.columns
            
            if has_labels:
                y_true = test_data['income']
                X_data = test_data.drop('income', axis=1)
            else:
                y_true = None
                X_data = test_data
            
            # Load model
            model = load_model(model_name)
            
            if model is not None and st.button("🚀 Make Predictions", type="primary"):
                with st.spinner("Making predictions..."):
                    try:
                        # Make predictions
                        if model_name in ['K-Nearest Neighbors', 'Logistic Regression', 'Naive Bayes']:
                            X_scaled = scaler.transform(X_data)
                            predictions = model.predict(X_scaled)
                            pred_proba = model.predict_proba(X_scaled)[:, 1]
                        else:
                            predictions = model.predict(X_data)
                            pred_proba = model.predict_proba(X_data)[:, 1]
                        
                        st.success("✅ Predictions completed!")
                        
                        # Display results
                        results = X_data.copy()
                        results['Predicted_Income'] = ['<=50K' if p == 0 else '>50K' for p in predictions]
                        results['Probability_>50K'] = pred_proba
                        
                        if has_labels:
                            results['Actual_Income'] = ['<=50K' if y == 0 else '>50K' for y in y_true]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Predictions", len(predictions))
                            st.metric("Predicted <=50K", int(sum(predictions == 0)))
                        with col2:
                            st.metric("Predicted >50K", int(sum(predictions == 1)))
                            st.metric("Avg Probability >50K", f"{pred_proba.mean():.4f}")
                        
                        # If we have labels, show metrics
                        if has_labels:
                            st.markdown("### 📊 Model Performance")
                            
                            accuracy = accuracy_score(y_true, predictions)
                            auc = roc_auc_score(y_true, pred_proba)
                            precision = precision_score(y_true, predictions)
                            recall = recall_score(y_true, predictions)
                            f1 = f1_score(y_true, predictions)
                            mcc = matthews_corrcoef(y_true, predictions)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Accuracy", f"{accuracy:.4f}")
                                st.metric("AUC", f"{auc:.4f}")
                            with col2:
                                st.metric("Precision", f"{precision:.4f}")
                                st.metric("Recall", f"{recall:.4f}")
                            with col3:
                                st.metric("F1 Score", f"{f1:.4f}")
                                st.metric("MCC", f"{mcc:.4f}")
                            
                            # Confusion Matrix
                            st.markdown("### 📊 Confusion Matrix")
                            cm = confusion_matrix(y_true, predictions)
                            
                            fig = px.imshow(cm, 
                                           labels=dict(x="Predicted", y="Actual", color="Count"),
                                           x=['<=50K', '>50K'],
                                           y=['<=50K', '>50K'],
                                           color_continuous_scale="Blues",
                                           text_auto=True)
                            fig.update_layout(title=f"Confusion Matrix - {model_name}")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Classification Report
                            st.markdown("### 📋 Classification Report")
                            report = classification_report(y_true, predictions, 
                                                          target_names=['<=50K', '>50K'],
                                                          output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df, use_container_width=True)
                        
                        # Show predictions
                        st.markdown("### 📊 Prediction Results")
                        st.dataframe(results.head(20))
                        
                        # Download predictions
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name=f"predictions_{model_name.replace(' ', '_')}.csv",
                            mime='text/csv'
                        )
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
        
        else:
            st.info("👆 Please upload a CSV file to make predictions")
            
            # Option to use pre-loaded test data
            if st.checkbox("Use pre-loaded test data for demonstration"):
                if X_test_full is not None and y_test_full is not None:
                    st.success(f"✅ Using pre-loaded test data! Shape: {X_test_full.shape}")
                    
                    # Sample data
                    sample_size = st.slider("Select number of samples:", 10, min(100, len(X_test_full)), 50)
                    X_sample = X_test_full.head(sample_size)
                    y_sample = y_test_full.head(sample_size).values.ravel()
                    
                    model = load_model(model_name)
                    
                    if model is not None and st.button("🚀 Make Predictions on Sample", type="primary"):
                        with st.spinner("Making predictions..."):
                            try:
                                # Make predictions
                                if model_name in ['K-Nearest Neighbors', 'Logistic Regression', 'Naive Bayes']:
                                    X_scaled = scaler.transform(X_sample)
                                    predictions = model.predict(X_scaled)
                                    pred_proba = model.predict_proba(X_scaled)[:, 1]
                                else:
                                    predictions = model.predict(X_sample)
                                    pred_proba = model.predict_proba(X_sample)[:, 1]
                                
                                # Calculate metrics
                                accuracy = accuracy_score(y_sample, predictions)
                                precision = precision_score(y_sample, predictions)
                                recall = recall_score(y_sample, predictions)
                                f1 = f1_score(y_sample, predictions)
                                
                                st.success("✅ Predictions completed!")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Accuracy", f"{accuracy:.4f}")
                                with col2:
                                    st.metric("Precision", f"{precision:.4f}")
                                with col3:
                                    st.metric("Recall", f"{recall:.4f}")
                                with col4:
                                    st.metric("F1 Score", f"{f1:.4f}")
                                
                                # Confusion Matrix
                                st.markdown("### 📊 Confusion Matrix")
                                cm = confusion_matrix(y_sample, predictions)
                                
                                fig = px.imshow(cm, 
                                               labels=dict(x="Predicted", y="Actual", color="Count"),
                                               x=['<=50K', '>50K'],
                                               y=['<=50K', '>50K'],
                                               color_continuous_scale="Blues",
                                               text_auto=True)
                                fig.update_layout(title=f"Confusion Matrix - {model_name}")
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error during prediction: {e}")

# =========================================
# Page 3: Model Comparison
# =========================================

elif page == "📉 Model Comparison":
    st.markdown('<p class="sub-header">⚖️ Detailed Model Comparison</p>', unsafe_allow_html=True)
    
    if results_df is not None:
        # Select models to compare
        selected_models = st.multiselect(
            "Select models to compare:",
            results_df['Model'].tolist(),
            default=results_df['Model'].tolist()[:3]
        )
        
        if len(selected_models) > 0:
            # Filter data
            comparison_df = results_df[results_df['Model'].isin(selected_models)]
            
            # Metric selection
            metric = st.selectbox("Select metric to compare:", 
                                 ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'])
            
            # Bar chart
            fig = px.bar(comparison_df, x='Model', y=metric, 
                        color='Model',
                        title=f"{metric} Comparison",
                        labels={'Model': 'Model', metric: metric})
            st.plotly_chart(fig, use_container_width=True)
            
            # Side-by-side comparison
            st.markdown("### 📊 Side-by-Side Metric Comparison")
            
            metrics_to_show = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
            
            fig = go.Figure()
            for model in selected_models:
                model_data = comparison_df[comparison_df['Model'] == model]
                fig.add_trace(go.Bar(
                    name=model,
                    x=metrics_to_show,
                    y=[model_data[m].values[0] for m in metrics_to_show]
                ))
            
            fig.update_layout(
                barmode='group',
                title="All Metrics Comparison",
                xaxis_title="Metrics",
                yaxis_title="Score",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical summary
            st.markdown("### 📈 Statistical Summary")
            summary_stats = comparison_df[['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']].describe()
            st.dataframe(summary_stats, use_container_width=True)
            
        else:
            st.warning("⚠️ Please select at least one model to compare")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🎓 Machine Learning Assignment 2 | M.Tech (AIML/DSE) | BITS Pilani</p>
        <p>Built with Streamlit 🎈 | Powered by scikit-learn</p>
    </div>
""", unsafe_allow_html=True)
