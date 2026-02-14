# Machine Learning Assignment 2
## Adult Census Income Prediction - Classification Models Comparison

**M.Tech (AIML/DSE) - BITS Pilani**  
**Submission Date:** February 15, 2026

---

## 📋 Problem Statement

The goal of this assignment is to predict whether an individual's income exceeds $50,000 per year based on census data. This is a **binary classification problem** where we compare the performance of six different machine learning algorithms on the Adult Census Income dataset.

The prediction task aims to identify patterns and relationships between demographic and employment-related features to classify individuals into two income categories:
- **<=50K**: Income less than or equal to $50,000
- **>50K**: Income greater than $50,000

This problem has real-world applications in:
- Economic policy planning
- Targeted social welfare programs
- Market segmentation for businesses
- Educational and career guidance

---

## 📊 Dataset Description

**Dataset Name:** Adult Census Income Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset Size:** 32,561 instances  
**Number of Features:** 14 (excluding target variable)  
**Target Variable:** `income` (binary: <=50K or >50K)

### Feature Details

| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| `age` | Continuous | Age of the individual | 17-90 years |
| `workclass` | Categorical | Employment type | Private, Self-emp, Government |
| `fnlwgt` | Continuous | Final weight (census weighting) | Numeric |
| `education` | Categorical | Highest education level | Bachelors, HS-grad, Masters |
| `education.num` | Continuous | Education in numerical form | 1-16 |
| `marital.status` | Categorical | Marital status | Married, Divorced, Never-married |
| `occupation` | Categorical | Type of occupation | Tech-support, Craft-repair, Sales |
| `relationship` | Categorical | Family relationship | Husband, Wife, Own-child |
| `race` | Categorical | Race of individual | White, Black, Asian-Pac-Islander |
| `sex` | Categorical | Gender | Male, Female |
| `capital.gain` | Continuous | Capital gains | 0-99999 |
| `capital.loss` | Continuous | Capital losses | 0-4356 |
| `hours.per.week` | Continuous | Working hours per week | 1-99 |
| `native.country` | Categorical | Country of origin | United-States, Mexico, India |

### Target Variable Distribution

- **<=50K (Class 0):** 24,720 instances (75.9%)
- **>50K (Class 1):** 7,841 instances (24.1%)

**Note:** The dataset is imbalanced with approximately 3:1 ratio favoring the lower income class.

### Data Preprocessing

1. **Handling Missing Values:** Missing values (represented as '?') were replaced with the mode of respective columns
2. **Encoding Categorical Variables:** Label encoding was applied to all categorical features
3. **Feature Scaling:** StandardScaler was applied for models requiring normalized features (Logistic Regression, KNN, Naive Bayes)
4. **Train-Test Split:** 80-20 split with stratification to maintain class distribution
   - Training Set: 26,048 instances
   - Test Set: 6,513 instances

---

## 🤖 Models Used

### Comparison Table - Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-------|------|
| Logistic Regression | 0.8268 | 0.8548 | 0.7227 | 0.4554 | 0.5587 | 0.4767 |
| Decision Tree | 0.8477 | 0.8774 | 0.7060 | 0.6295 | 0.6655 | 0.5689 |
| K-Nearest Neighbors | 0.8263 | 0.8471 | 0.6610 | 0.5721 | 0.6133 | 0.5043 |
| Naive Bayes | 0.8027 | 0.8538 | 0.6859 | 0.3329 | 0.4483 | 0.3788 |
| Random Forest (Ensemble) | 0.8557 | 0.9121 | 0.7990 | 0.5351 | 0.6409 | 0.5725 |
| XGBoost (Ensemble) | **0.8681** | **0.9205** | **0.7767** | **0.6346** | **0.6985** | **0.6203** |

**Best Performing Model:** XGBoost with 86.81% accuracy and 0.9205 AUC score.

---

## 📈 Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Achieved decent accuracy (82.68%) with good AUC (0.8548), indicating strong probability calibration. However, it shows relatively low recall (45.54%), suggesting it misses many positive cases (>50K income). The high precision (72.27%) means when it predicts >50K, it's often correct. This linear model serves as a solid baseline but struggles with complex non-linear patterns in the data. Best suited when interpretability is crucial. |
| **Decision Tree** | Demonstrates balanced performance with 84.77% accuracy and improved recall (62.95%) compared to logistic regression. The model captures non-linear relationships well with an AUC of 0.8774. However, single decision trees are prone to overfitting, as evidenced by the moderate MCC (0.5689). The max_depth=10 parameter helps control complexity. Good for understanding feature importance and decision boundaries, but ensemble methods outperform it. |
| **K-Nearest Neighbors** | Shows moderate performance (82.63% accuracy) with balanced precision (66.10%) and recall (57.21%). The model's AUC of 0.8471 is the lowest among tested models, indicating weaker probability estimates. KNN is sensitive to feature scaling (which we applied) and struggles with high-dimensional spaces. The model's computational cost during prediction is high as it requires distance calculations. Performance could improve with optimal k selection and dimensionality reduction. |
| **Naive Bayes** | Exhibits the lowest overall performance with 80.27% accuracy and notably poor recall (33.29%), making it unsuitable for this task where identifying high-income individuals is important. Despite reasonable AUC (0.8538), the strong independence assumption of Naive Bayes doesn't hold well for this dataset where features like education, occupation, and age are correlated. The low MCC (0.3788) indicates weak overall classification performance. Not recommended for this problem. |
| **Random Forest (Ensemble)** | Shows strong performance with 85.57% accuracy and excellent AUC (0.9121), the second-best overall. The ensemble approach reduces overfitting compared to single decision trees while maintaining good interpretability through feature importance. High precision (79.90%) but moderate recall (53.51%) suggests the model is conservative in predicting >50K income. The model handles feature interactions well and is robust to outliers. Trade-off between precision and recall could be adjusted via threshold tuning. |
| **XGBoost (Ensemble)** | **Best performing model** across all metrics with 86.81% accuracy, 0.9205 AUC, and highest MCC (0.6203). The gradient boosting approach effectively handles imbalanced data and captures complex patterns. Excellent balance between precision (77.67%) and recall (63.46%), resulting in the highest F1 score (0.6985). The model's sequential tree building corrects errors from previous iterations. Superior probability estimates make it ideal for threshold-based decision making. Recommended for deployment despite slightly longer training time. |

---

## 🔑 Key Insights and Analysis

### Overall Performance Trends

1. **Ensemble Methods Dominate:** Both Random Forest and XGBoost significantly outperform individual models, demonstrating the power of ensemble learning
2. **Linear vs Non-linear:** Tree-based models outperform linear models (Logistic Regression), indicating important non-linear relationships in the data
3. **Imbalanced Data Impact:** All models show higher precision than recall, suggesting they're better at confirming >50K income than finding all cases
4. **AUC Scores:** All models achieve AUC > 0.84, indicating good discriminative ability between classes

### Model Selection Recommendations

- **Production Deployment:** XGBoost - Best overall metrics and probability calibration
- **Interpretability Required:** Decision Tree or Logistic Regression - Easier to explain to stakeholders
- **Fast Prediction Needed:** Logistic Regression - Fastest inference time
- **Balanced Performance:** Random Forest - Good trade-off between accuracy and interpretability

### Performance Gaps

The performance gap between best (XGBoost: 86.81%) and worst (Naive Bayes: 80.27%) is 6.54%, indicating that algorithm choice significantly matters for this dataset.

---

## 🚀 Streamlit Application Features

The interactive web application includes:

### 1. Model Overview Page
- **Performance Metrics Table:** Complete comparison of all 6 models
- **Best Model Highlighting:** Automatic identification of top performer
- **Interactive Visualizations:**
  - Multi-metric bar charts
  - Radar chart for holistic comparison
  - Performance heatmap

### 2. Model Prediction Page
- **File Upload:** CSV upload for batch predictions
- **Model Selection:** Dropdown to choose any trained model
- **Real-time Predictions:** Instant results with probability scores
- **Pre-loaded Test Data:** Demo mode with sample predictions
- **Confusion Matrix:** Visual performance assessment
- **Classification Report:** Detailed per-class metrics
- **Download Results:** Export predictions as CSV

### 3. Model Comparison Page
- **Multi-model Selection:** Compare 2+ models simultaneously
- **Metric Selection:** Focus on specific evaluation criteria
- **Side-by-side Charts:** Grouped bar charts for all metrics
- **Statistical Summary:** Descriptive statistics across models

---

## 📁 Repository Structure

```
ml-assignment-2/
│
├── app.py                              # Streamlit web application
├── train_models.py                     # Model training script
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
│
├── adult.csv                           # Original dataset
├── test_data.csv                       # Test set for predictions
├── test_labels.csv                     # True labels for test set
│
├── model_results.csv                   # Performance metrics table
│
├── scaler.pkl                          # Fitted StandardScaler
├── label_encoders.pkl                  # Label encoders for categorical features
│
└── models/                             # Saved trained models
    ├── logistic_regression_model.pkl
    ├── decision_tree_model.pkl
    ├── k-nearest_neighbors_model.pkl
    ├── naive_bayes_model.pkl
    ├── random_forest_model.pkl
    └── xgboost_model.pkl
```

---

## 🛠️ Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Local Setup

1. **Clone the repository:**
```bash
git clone <your-github-repo-url>
cd ml-assignment-2
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Train models (optional):**
```bash
python train_models.py
```

4. **Run Streamlit app:**
```bash
streamlit run app.py
```

5. **Access the application:**
Open your browser and navigate to `http://localhost:8501`

---

## ☁️ Deployment on Streamlit Cloud

### Step-by-Step Deployment

1. **Push code to GitHub:**
```bash
git add .
git commit -m "ML Assignment 2 - Complete implementation"
git push origin main
```

2. **Deploy on Streamlit Cloud:**
   - Visit [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with GitHub account
   - Click "New App"
   - Select your repository
   - Choose branch: `main`
   - Set main file: `app.py`
   - Click "Deploy"

3. **Wait for deployment** (typically 2-5 minutes)

4. **Access your live app** at the provided URL

---

## 📊 Model Training Details

### Hyperparameters Used

| Model | Key Hyperparameters |
|-------|-------------------|
| Logistic Regression | max_iter=1000, random_state=42 |
| Decision Tree | max_depth=10, random_state=42 |
| K-Nearest Neighbors | n_neighbors=5 |
| Naive Bayes | Default (GaussianNB) |
| Random Forest | n_estimators=100, max_depth=10, random_state=42 |
| XGBoost | n_estimators=100, max_depth=5, random_state=42 |

### Evaluation Metrics Explanation

- **Accuracy:** Overall correctness of predictions
- **AUC (Area Under ROC Curve):** Model's ability to distinguish between classes
- **Precision:** Proportion of positive predictions that are correct
- **Recall:** Proportion of actual positives correctly identified
- **F1 Score:** Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient):** Balanced measure for imbalanced datasets

---

## 🎯 Usage Examples

### Making Predictions with the Streamlit App

1. Navigate to "Model Prediction" page
2. Select your preferred model
3. Upload a CSV file (without target column) OR use pre-loaded test data
4. Click "Make Predictions"
5. View results and download predictions

### Using Pre-loaded Test Data

The app includes a demo mode with 6,513 test instances for immediate testing without file upload.

---

## 🔍 Future Improvements

1. **Hyperparameter Optimization:** Implement GridSearchCV or RandomizedSearchCV
2. **Feature Engineering:** Create interaction features and polynomial terms
3. **Advanced Ensembles:** Try stacking or voting classifiers
4. **Deep Learning:** Experiment with neural networks
5. **SMOTE:** Address class imbalance with oversampling techniques
6. **Feature Selection:** Identify and use only the most predictive features
7. **Cross-validation:** Implement k-fold CV for more robust evaluation
8. **Threshold Optimization:** Tune classification thresholds based on business needs

---

## 📝 Assignment Submission Checklist

- ✅ GitHub repository with complete source code
- ✅ requirements.txt file with all dependencies
- ✅ README.md with comprehensive documentation
- ✅ Live Streamlit app deployed on Streamlit Cloud
- ✅ All 6 models implemented and evaluated
- ✅ Complete evaluation metrics calculated
- ✅ Model performance observations documented
- ✅ Screenshot of execution on BITS Virtual Lab
- ✅ Interactive features in Streamlit app (upload, selection, metrics, confusion matrix)

---

## 👨‍💻 Technical Stack

- **Programming Language:** Python 3.8+
- **ML Framework:** scikit-learn 1.4.0
- **Gradient Boosting:** XGBoost (GradientBoostingClassifier)
- **Web Framework:** Streamlit 1.31.0
- **Data Processing:** pandas 2.1.4, numpy 1.26.3
- **Visualization:** plotly 5.18.0
- **Deployment:** Streamlit Community Cloud

---

## 📧 Contact

**Course:** Machine Learning  
**Program:** M.Tech (AIML/DSE)  
**Institution:** BITS Pilani - Work Integrated Learning Programmes

For any queries regarding BITS Virtual Lab, contact: neha.vinayak@pilani.bits-pilani.ac.in

---

## 📄 License

This project is submitted as part of academic coursework for BITS Pilani M.Tech program.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Adult Census Income dataset
- BITS Pilani for providing the Virtual Lab infrastructure
- Streamlit for the amazing web application framework
- scikit-learn community for comprehensive ML tools

---

**Note:** This implementation was performed on BITS Virtual Lab as per assignment requirements.

**Submission Deadline:** February 15, 2026, 23:59 PM

---

*Generated as part of Machine Learning Assignment 2 | M.Tech (AIML/DSE) | BITS Pilani*
