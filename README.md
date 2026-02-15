# Machine Learning Assignment 2
## Adult Census Income Prediction - Classification Models Comparison

**M.Tech (AIML/DSE) - BITS Pilani**  
**Submission Date:** February 15, 2026  

---

## 📋 Problem Statement

The objective of this project is to predict whether an individual's annual income exceeds $50,000 per year based on demographic and employment-related attributes such as age, education, occupation, and working hours.

This is a **binary classification problem** where we compare the performance of six different machine learning algorithms on the Adult Census Income dataset.

The prediction task aims to identify patterns and relationships between demographic features to classify individuals into two income categories:
- **<=50K**: Income less than or equal to $50,000
- **>50K**: Income greater than $50,000

### Real-World Applications:
- Economic policy planning and implementation
- Targeted social welfare programs
- Market segmentation for businesses
- Educational and career guidance services
- Demographic analysis for government agencies

---

## 📊 Dataset Description

**Dataset Name:** Adult Census Income Dataset  
**Source:** UCI Machine Learning Repository (via Kaggle)  
**Dataset Size:** 32,561 instances  
**Number of Features:** 14 (excluding target variable)  
**Target Variable:** `income` (binary: <=50K or >50K)

### Feature Details

| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| `age` | Continuous | Age of the individual | 17-90 years |
| `workclass` | Categorical | Employment type | Private, Self-emp-not-inc, Local-gov, State-gov, Federal-gov |
| `fnlwgt` | Continuous | Final sampling weight (census weighting) | Numeric values |
| `education` | Categorical | Highest education level achieved | Bachelors, HS-grad, Masters, Doctorate, Some-college |
| `education.num` | Continuous | Education level in numerical form | 1-16 |
| `marital.status` | Categorical | Marital status | Married-civ-spouse, Divorced, Never-married, Separated |
| `occupation` | Categorical | Type of occupation | Tech-support, Craft-repair, Sales, Exec-managerial |
| `relationship` | Categorical | Family relationship status | Husband, Wife, Own-child, Not-in-family |
| `race` | Categorical | Race of individual | White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo |
| `sex` | Categorical | Gender | Male, Female |
| `capital.gain` | Continuous | Capital gains income | 0-99999 |
| `capital.loss` | Continuous | Capital losses | 0-4356 |
| `hours.per.week` | Continuous | Working hours per week | 1-99 |
| `native.country` | Categorical | Country of origin | United-States, Mexico, India, Philippines, etc. |

### Target Variable Distribution

- **<=50K (Class 0):** ~75% of instances
- **>50K (Class 1):** ~25% of instances

**Note:** The dataset exhibits class imbalance with approximately 3:1 ratio favoring the lower income class.

### Data Preprocessing Steps

1. **Handling Missing Values:** 
   - Missing values represented as '?' or ' ?' were identified
   - Rows with missing values were removed to ensure data quality

2. **Data Cleaning:**
   - Stripped whitespace from all object (string) columns
   - Ensured consistent formatting across categorical variables

3. **Encoding Target Variable:**
   - Binary mapping: `<=50K` → 0, `>50K` → 1

4. **Feature Engineering:**
   - Applied one-hot encoding to all categorical variables
   - Used `drop_first=True` to avoid multicollinearity

5. **Feature Scaling:**
   - Applied StandardScaler to normalize features
   - Used only for models sensitive to feature scales (Logistic Regression, KNN, Naive Bayes)

6. **Train-Test Split:**
   - 80-20 split ratio with stratification
   - Random state: 42 (for reproducibility)

---

## 🤖 Models Used

### Comparison Table - Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-------|------|
| Logistic Regression | 0.8548 | 0.9132 | 0.7504 | 0.6245 | 0.6817 | 0.5928 |
| Decision Tree | 0.8583 | 0.8990 | 0.7870 | 0.5905 | 0.6748 | 0.5964 |
| K-Nearest Neighbors | 0.8258 | 0.8497 | 0.6667 | 0.6005 | 0.6319 | 0.5194 |
| Naive Bayes | 0.5488 | 0.7761 | 0.3514 | 0.9601 | 0.5144 | 0.3454 |
| Random Forest (Ensemble) | 0.8550 | 0.9104 | 0.7467 | 0.6318 | 0.6845 | 0.5946 |
| XGBoost (Ensemble) | **0.8722** | **0.9338** | **0.7903** | **0.6625** | **0.7208** | **0.6429** |

**🏆 Best Performing Model:** XGBoost (Gradient Boosting) with 87.22% accuracy and 0.9338 AUC score.

---

## 📈 Observations on Model Performance

### Logistic Regression
**Performance Metrics:**
- Accuracy: 85.48% | AUC: 0.9132 | Precision: 75.04% | Recall: 62.45% | F1: 0.6817

**Observations:**
Logistic Regression demonstrates strong performance with 85.48% accuracy and excellent AUC (0.9132), indicating superior probability calibration and ranking ability. The model achieves a good balance between precision (75.04%) and recall (62.45%), with an F1 score of 0.6817. This linear model provides excellent interpretability through coefficient analysis while maintaining competitive performance. The high AUC suggests the model effectively distinguishes between income classes and produces well-calibrated probability estimates. The MCC score of 0.5928 confirms strong overall classification performance even with class imbalance. Best suited for applications requiring transparent decision-making processes and when stakeholders need to understand the contribution of each feature.

**Strengths:** Fast training/prediction, highly interpretable coefficients, excellent probability estimates, low computational requirements, strong AUC performance
**Weaknesses:** Cannot capture complex non-linear relationships, assumes linear separability, limited feature interaction modeling

**Recommendation:** Excellent baseline model with strong performance. Use when interpretability is critical or as an ensemble component.

---

### Decision Tree
**Performance Metrics:**
- Accuracy: 85.83% | AUC: 0.8990 | Precision: 78.70% | Recall: 59.05% | F1: 0.6748

**Observations:**
Decision Tree achieves the second-highest accuracy at 85.83% with the highest precision (78.70%) among all models, indicating excellent reliability when predicting high-income individuals. However, the recall of 59.05% suggests the model is conservative in making positive predictions. The AUC of 0.8990 demonstrates good discriminative ability, though slightly lower than ensemble methods. The model excels at capturing non-linear relationships and feature interactions through its hierarchical structure. With max_depth=10, the tree provides good balance between model complexity and generalization. The MCC score of 0.5964 is competitive, showing robust performance considering class imbalance. Decision trees offer intuitive visualization of decision paths and reveal important feature interactions naturally.

**Strengths:** Highest precision (78.70%), handles non-linear relationships, interpretable structure, captures feature interactions automatically, no feature scaling needed
**Weaknesses:** Prone to overfitting, high variance, unstable with small data changes, lower recall compared to ensemble methods

**Recommendation:** Use when precision is more important than recall, or when interpretable decision rules are required.

---

### K-Nearest Neighbors (KNN)
**Performance Metrics:**
- Accuracy: 82.58% | AUC: 0.8497 | Precision: 66.67% | Recall: 60.05% | F1: 0.6319

**Observations:**
KNN demonstrates moderate performance with 82.58% accuracy and the most balanced precision-recall trade-off (66.67% precision, 60.05% recall). The model's AUC of 0.8497 is the lowest among traditional classifiers, indicating weaker probability estimates and ranking capability. As a non-parametric, instance-based learner, KNN stores all training data, making it memory-intensive and computationally expensive during prediction. The algorithm is highly sensitive to feature scaling (which was applied) and struggles with high-dimensional spaces due to the curse of dimensionality. With n_neighbors=5, the model captures local patterns well but may miss global trends. The MCC score of 0.5194 reflects adequate but not exceptional performance.

**Strengths:** No training phase, naturally multi-class ready, adapts to local patterns, simple implementation, most balanced precision-recall
**Weaknesses:** Lowest accuracy (82.58%), slowest predictions, memory intensive, sensitive to irrelevant features, poor scalability, lowest AUC

**Recommendation:** Consider for smaller datasets or when local pattern matching is important. Not recommended for production at scale.

---

### Naive Bayes
**Performance Metrics:**
- Accuracy: 54.88% | AUC: 0.7761 | Precision: 35.14% | Recall: 96.01% | F1: 0.5144

**Observations:**
Naive Bayes exhibits dramatically different behavior compared to other models, achieving extremely high recall (96.01%) but very low precision (35.14%) and the lowest accuracy (54.88%). This extreme recall suggests the model predicts ">50K" for most instances, capturing nearly all high-income individuals but with many false positives. The AUC of 0.7761 indicates fair discriminative ability, but the severe precision-recall imbalance makes it unsuitable for most applications. The strong independence assumption of Naive Bayes fails for this dataset where features like education, occupation, and age are clearly correlated. The very low MCC (0.3454) confirms poor overall classification quality despite high recall. This model demonstrates the classic precision-recall trade-off taken to an extreme.

**Strengths:** Captures 96.01% of high-income individuals (highest recall), fast training and prediction, simple probabilistic framework
**Weaknesses:** Lowest accuracy (54.88%), lowest precision (35.14%), produces excessive false positives, independence assumption violated, lowest MCC

**Recommendation:** NOT RECOMMENDED for deployment. Only consider if missing even a single high-income individual is catastrophically expensive.

---

### Random Forest (Ensemble)
**Performance Metrics:**
- Accuracy: 85.50% | AUC: 0.9104 | Precision: 74.67% | Recall: 63.18% | F1: 0.6845

**Observations:**
Random Forest achieves excellent performance with 85.50% accuracy and strong AUC (0.9104), demonstrating the power of ensemble learning. The model provides good balance between precision (74.67%) and recall (63.18%), resulting in a competitive F1 score of 0.6845. As an ensemble of 100 decision trees, Random Forest effectively reduces overfitting compared to single decision trees while maintaining interpretability through feature importance analysis. The model handles feature interactions naturally, is robust to outliers and missing values, and provides reliable probability estimates. With max_depth and n_estimators parameters, the model achieves good generalization. The MCC score of 0.5946 confirms strong performance considering class imbalance.

**Strengths:** Strong AUC (0.9104), reduces overfitting, provides feature importance, robust to outliers, parallelizable, good generalization
**Weaknesses:** Less interpretable than single tree, larger model size, slightly slower predictions, computational intensity

**Recommendation:** Excellent choice for production deployment. Strong all-around performance with good reliability.

---

### XGBoost (Ensemble) - 🏆 Best Model
**Performance Metrics:**
- Accuracy: 87.22% | AUC: 0.9338 | Precision: 79.03% | Recall: 66.25% | F1: 0.7208

**Observations:**
**XGBoost emerges as the clear winner** with the highest performance across all key metrics: 87.22% accuracy, 0.9338 AUC, 79.03% precision, 66.25% recall, and 0.7208 F1 score. The exceptional AUC of 0.9338 demonstrates superior ranking ability and probability calibration. XGBoost's gradient boosting approach sequentially builds trees where each tree corrects errors from previous iterations, effectively handling complex patterns and the class imbalance. The model achieves the best precision-recall balance, with the highest F1 score among all classifiers. Built-in regularization (L1/L2) prevents overfitting, while features like early stopping and tree pruning enhance generalization. The highest MCC score (0.6429) confirms this is the most reliable classifier for the dataset.

**Strengths:** Best accuracy (87.22%), highest AUC (0.9338), best F1 score (0.7208), highest MCC (0.6429), handles imbalanced data excellently, built-in regularization, early stopping capability, superior probability estimates
**Weaknesses:** More hyperparameters to tune, longer training time, requires more computational resources, less interpretable than simple models

**Recommendation:** **STRONGLY RECOMMENDED for production deployment.** Highest performance justifies any additional computational cost. This is the model to deploy.

---

## 🔑 Key Insights and Analysis

### Overall Performance Trends

1. **XGBoost Dominates:** 
   - Clear winner with 87.22% accuracy and 0.9338 AUC
   - 1.39% accuracy improvement over second-best (Decision Tree: 85.83%)
   - Superior across ALL evaluation metrics

2. **Strong Traditional Classifiers:**
   - Logistic Regression (85.48%), Decision Tree (85.83%), and Random Forest (85.50%) all perform comparably
   - All three achieve >85% accuracy
   - Demonstrates multiple viable approaches to the problem

3. **Ensemble Methods Excel:**
   - Top 3 models are ensemble or boosting methods (XGBoost, Random Forest)
   - Plus strong performance from Logistic Regression
   - Confirms value of model combination and sequential learning

4. **Naive Bayes Outlier:**
   - Dramatically different behavior (54.88% accuracy, 96.01% recall)
   - Extreme precision-recall imbalance
   - Demonstrates failed independence assumption
   - Unsuitable for this application

5. **AUC Performance:**
   - XGBoost leads with exceptional 0.9338 AUC
   - All models except Naive Bayes achieve AUC > 0.84
   - Strong discriminative ability across classifiers

6. **Precision vs Recall Trade-off:**
   - Decision Tree: Highest precision (78.70%), moderate recall (59.05%)
   - Naive Bayes: Highest recall (96.01%), lowest precision (35.14%)
   - XGBoost: Best balanced performance (79.03% precision, 66.25% recall)

### Model Selection Recommendations

**For Different Use Cases:**

- 🥇 **Production Deployment:** **XGBoost**
  - Best overall performance (87.22% accuracy, 0.9338 AUC)
  - Highest F1 score (0.7208) and MCC (0.6429)
  - **STRONGLY RECOMMENDED** for real-world deployment

- 📊 **When Interpretability is Critical:** Logistic Regression
  - Strong performance (85.48% accuracy)
  - Clear coefficient interpretation
  - Regulatory compliance / explainable AI requirements

- 🎯 **When Precision Matters Most:** Decision Tree
  - Highest precision (78.70%)
  - Minimizes false positives
  - Use when false positive cost is high (e.g., loan approval)

- 🔍 **When Recall Matters Most:** XGBoost (balanced) or Naive Bayes (extreme)
  - XGBoost: 66.25% recall with good precision
  - Naive Bayes: 96.01% recall but poor precision
  - XGBoost preferred unless missing positives is catastrophic

- ⚡ **Fast Prediction Required:** Logistic Regression
  - Fastest inference time
  - Minimal computational requirements
  - Still maintains 85.48% accuracy

- ⚖️ **Balanced Performance:** Random Forest
  - Strong all-around metrics
  - Robust and reliable
  - Good fallback if XGBoost unavailable

### Performance Gaps

**Accuracy Range:** 54.88% (Naive Bayes) to 87.22% (XGBoost) = 32.34% spread
**Competitive Range:** 82.58% (KNN) to 87.22% (XGBoost) = 4.64% spread

Excluding Naive Bayes outlier, the 4.64% gap shows meaningful performance differences. The 1.39% improvement from second-best to best (Decision Tree 85.83% → XGBoost 87.22%) represents correctly classifying ~84 additional individuals in a 6,000-person test set.

---

## 🚀 Streamlit Application Features

The interactive web application provides comprehensive model exploration and prediction capabilities:

### 1. 📈 Model Overview Page
- **Performance Metrics Table:** Complete comparison of all 6 models
- **Best Model Highlighting:** XGBoost automatically identified as top performer
- **Interactive Visualizations:**
  - Multi-metric bar charts
  - Radar chart for 360° performance view
  - Performance heatmap

### 2. 🔍 Model Prediction Page
- **File Upload:** CSV upload for batch predictions
- **Model Selection:** Choose any of the 6 trained models
- **Real-time Predictions:** Instant results with probability scores
- **Pre-loaded Test Data:** Demo mode with 100 sample predictions
- **Performance Visualization:**
  - Confusion matrix
  - Classification report
  - All 6 evaluation metrics
- **Download Results:** Export predictions as CSV

### 3. 📉 Model Comparison Page
- **Multi-model Selection:** Compare 2+ models simultaneously
- **Metric Selection:** Focus on specific criteria
- **Side-by-side Charts:** Grouped bar charts for comparison
- **Statistical Summary:** Descriptive statistics across models

---

## 📁 Repository Structure

```
ml-assignment-2/
│
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── README.md                           # This comprehensive documentation
├── test_data.csv                       # Test set (100 samples for demo)
│
└── model/                              # Saved trained models
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── xgboost.pkl
    ├── scaler.pkl                      # Fitted StandardScaler
    ├── train_models.py                 # Model training script  
    ├── model_results.csv               # Performance comparison table
    

---

## 📊 Model Training Details

### Hyperparameters Used

| Model | Key Hyperparameters |
|-------|-------------------|
| Logistic Regression | max_iter=1000 |
| Decision Tree | max_depth=10, random_state=42 |
| K-Nearest Neighbors | n_neighbors=5 |
| Naive Bayes | Default (GaussianNB) |
| Random Forest | n_estimators=100, random_state=42 |
| XGBoost | use_label_encoder=False, eval_metric='logloss', random_state=42 |

### Evaluation Metrics Explanation

**Accuracy (87.22% for XGBoost):**
- Overall correctness: (TP + TN) / Total
- Percentage of correct predictions

**AUC - Area Under ROC Curve (0.9338 for XGBoost):**
- Probability of correct ranking
- 0.9338 = Excellent discriminative ability
- Threshold-independent metric

**Precision (79.03% for XGBoost):**
- Of predictions as ">50K", how many are correct?
- Formula: TP / (TP + FP)
- High precision = Few false positives

**Recall (66.25% for XGBoost):**
- Of actual ">50K" cases, how many identified?
- Formula: TP / (TP + FN)
- High recall = Few missed opportunities

**F1 Score (0.7208 for XGBoost):**
- Harmonic mean of precision and recall
- Balances precision-recall trade-off

**MCC - Matthews Correlation Coefficient (0.6429 for XGBoost):**
- Most reliable metric for imbalanced data
- Range: -1 to +1
- 0.6429 = Strong positive correlation

---

## 🔍 Future Improvements

1. **Hyperparameter Optimization:**
   - GridSearchCV or RandomizedSearchCV
   - Bayesian optimization
   - Could improve XGBoost to ~88-89%

2. **Feature Engineering:**
   - Interaction features
   - Polynomial features
   - Domain-specific features

3. **Advanced Ensemble Methods:**
   - Stacking classifiers
   - Voting ensembles
   - Combine XGBoost + Random Forest

4. **Deep Learning:**
   - Neural networks
   - TabNet for tabular data

5. **Class Imbalance Techniques:**
   - SMOTE oversampling
   - Class weight adjustment
   - Could improve recall further

6. **Feature Selection:**
   - Recursive Feature Elimination
   - Mutual information
   - Reduce model complexity

7. **Cross-Validation:**
   - K-fold CV for robust estimates
   - Reduce variance in metrics

8. **Model Explainability:**
   - SHAP values
   - LIME
   - Partial Dependence Plots

---

## 👨‍💻 Technical Stack

- **Programming Language:** Python 3.9+
- **ML Framework:** scikit-learn
- **Gradient Boosting:** XGBoost
- **Web Framework:** Streamlit
- **Data Processing:** pandas, numpy
- **Visualization:** plotly

---

## 📧 Contact

**Course:** Machine Learning  
**Program:** M.Tech (AIML/DSE)  
**Institution:** BITS Pilani - Work Integrated Learning Programmes

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the Adult Census Income dataset
- **BITS Pilani** for Virtual Lab infrastructure
- **Streamlit** for the web application framework
- **scikit-learn and XGBoost communities** for ML tools

---

## 📊 Dataset Citation

```bibtex
@misc{adult_census_income_1996,
  author = "Ronny Kohavi and Barry Becker",
  title = "Adult Census Income",
  year = "1996",
  institution = "UCI Machine Learning Repository",
  url = "https://archive.ics.uci.edu/ml/datasets/adult"
}
```

---

## 🎯 Project Summary

This project successfully implements and compares six machine learning classification algorithms for predicting adult income levels. **XGBoost emerged as the clear winner with 87.22% accuracy and 0.9338 AUC**, demonstrating the effectiveness of gradient boosting for structured data prediction tasks.

**Key Achievements:**
- ✅ Implemented 6 diverse ML algorithms
- ✅ Achieved 87.22% accuracy (XGBoost)
- ✅ Excellent AUC of 0.9338
- ✅ F1 score of 0.7208 (highest among all models)
- ✅ Interactive web application deployed
- ✅ Complete documentation and reproducibility

**Best Model: XGBoost**
- **Accuracy:** 87.22% (highest)
- **AUC:** 0.9338 (highest)
- **Precision:** 79.03% (second highest)
- **Recall:** 66.25% (highest balanced recall)
- **F1 Score:** 0.7208 (highest)
- **MCC:** 0.6429 (highest)

**Recommendation:** Deploy XGBoost for production use.

---

**Note:** This implementation was performed on BITS Virtual Lab as per assignment requirements.

---

*Machine Learning Assignment 2 | M.Tech (AIML/DSE) | BITS Pilani*  
*Based on actual training results with XGBoost achieving 87.22% accuracy*
