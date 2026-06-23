# Anxiety Prediction
# 🧠 Anxiety Prediction and Mental Health Risk Analysis using Machine Learning

This project predicts an individual's **Anxiety Score** using behavioral, lifestyle, academic, and psychological factors. The objective is to identify the key contributors to anxiety and build a machine learning model capable of estimating anxiety levels based on user characteristics.

---

## 📌 Project Overview

Mental health issues, especially anxiety, have become increasingly prevalent among students and working professionals. This project applies Machine Learning techniques to analyze mental health data and predict anxiety scores using factors such as sleep quality, stress level, academic pressure, physical activity, and social support.

The project includes:

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Correlation Analysis
- Feature Engineering
- Regression Modeling
- Hyperparameter Tuning
- Feature Importance Analysis

---

## 📂 Dataset

The dataset contains demographic, lifestyle, academic, and psychological variables.

### Features

| Feature | Description |
|----------|-------------|
| age | Age of individual |
| gender | Gender |
| occupation | Occupation type |
| sleep_hours | Average daily sleep hours |
| sleep_quality | Sleep quality rating |
| social_media_hours | Daily social media usage |
| academic_work_pressure | Academic/work pressure level |
| physical_activity_days | Days of exercise per week |
| stress_level | Self-reported stress level |
| depression_score | Depression score |
| work_life_balance | Work-life balance rating |
| mood_score | Mood score |
| concentration_level | Concentration level |
| social_support | Social support rating |

### Target Variable

```python
anxiety_score
```

---

## 🛠️ Data Preprocessing

### Missing Value Handling

- Numerical Features → Median Imputation
- Categorical Features → Filled with `"Unknown"`

```python
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna("Unknown", inplace=True)
```

### Encoding

Categorical variables were transformed into numerical values using Label Encoding.

```python
LabelEncoder()
```

### Train-Test Split

```python
train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)
```

- Training Set: 80%
- Testing Set: 20%

---

## 📊 Exploratory Data Analysis

Performed:

- Dataset inspection
- Missing value analysis
- Duplicate detection
- Statistical summaries
- Correlation analysis
- Interactive visualization using D-Tale

### Correlation Findings

Positive Correlation with Anxiety:

| Feature | Correlation |
|----------|-------------|
| Stress Level | High Positive |
| Academic Work Pressure | High Positive |
| Depression Score | Positive |
| Social Media Hours | Positive |

Negative Correlation with Anxiety:

| Feature | Correlation |
|----------|-------------|
| Sleep Hours | Negative |
| Sleep Quality | Negative |
| Concentration Level | Negative |
| Work-Life Balance | Negative |
| Social Support | Negative |

---

## 🤖 Machine Learning Models

### 1. Linear Regression

Baseline regression model used for comparison.

```python
LinearRegression()
```

---

### 2. Random Forest Regressor

```python
RandomForestRegressor(
    n_estimators=500,
    max_depth=5
)
```

Advantages:

- Handles nonlinear relationships
- Robust against overfitting
- Provides feature importance

---

### 3. XGBoost Regressor

```python
XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05
)
```

Advantages:

- Gradient boosting framework
- Strong predictive performance
- Efficient handling of complex data patterns

---

## ⚙️ Hyperparameter Tuning

Used GridSearchCV with 5-Fold Cross Validation.

```python
GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid=param_grid,
    cv=5
)
```

Parameters tuned:

- n_estimators
- max_depth
- min_samples_split

---

## 📈 Model Evaluation

Evaluation Metrics:

### R² Score

Measures how well the model explains variance in anxiety scores.

```python
r2_score(y_test, predictions)
```

### Mean Absolute Error (MAE)

Measures average prediction error.

```python
mean_absolute_error(y_test, predictions)
```

---

## 🔍 Feature Importance Analysis

Random Forest Feature Importance was used to identify the most influential predictors.

Key Features:

1. Stress Level
2. Academic Work Pressure
3. Depression Score
4. Social Media Usage
5. Sleep Quality
6. Work-Life Balance
7. Concentration Level
8. Social Support

These factors showed the strongest relationship with anxiety prediction.

---

## 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- XGBoost
- D-Tale
- Jupyter Notebook

---

## 📁 Project Structure

```text
Anxiety-Prediction/
│
├── EDA.ipynb
├── anxiety_score.ipynb
├── clean_data.csv
├── mental_health_prediction.csv
├── README.md
│
└── outputs/
```

---

## 🎯 Key Learnings

- Data Cleaning and Preprocessing
- Exploratory Data Analysis
- Correlation Analysis
- Regression Modeling
- Ensemble Learning
- Hyperparameter Optimization
- Feature Importance Interpretation
- Mental Health Data Analytics

---

## 🚀 Future Improvements

- Streamlit Web Application
- SHAP Explainability
- Advanced Feature Engineering
- Model Deployment on Cloud
- Anxiety Risk Classification System
- Real-Time User Prediction Interface

---

## 👨‍💻 Author

**Amritesh Singh**

B.Tech, Production & Industrial Engineering  
Motilal Nehru National Institute of Technology Allahabad (MNNIT)

### Connect With Me

- GitHub: https://github.com/amriteshsingh956-tech
- LinkedIn: Add your LinkedIn profile here

---

⭐ If you found this project useful, consider giving it a star!
