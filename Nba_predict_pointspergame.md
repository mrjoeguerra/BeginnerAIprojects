## Preface

As part of my journey to deepen my understanding of artificial intelligence and machine learning, I set out to practice the complete end-to-end ML workflow using a real-world sports dataset. Basketball, and specifically the NBA, offers a rich source of structured data and well-defined prediction tasks, making it an ideal domain for applying and honing data science and machine learning skills.

With this project, my primary goal was to use NBA player statistics to build a predictive model for points per game (PTS). By doing so, I aimed to reinforce my grasp of the core processes and techniques that underpin successful AI/ML projects, including:

- Data loading, cleaning, and preprocessing
- Exploratory data analysis (EDA) and visualization
- Feature engineering and selection
- Model selection, training, and evaluation
- Interpretation of results and feature importance
- Model persistence and reproducibility

I chose this dataset because it allowed me to work with real, multi-dimensional data, explore regression modeling, and create visualizations that reveal insights about player performance. Throughout the project, I focused on writing clear, well-commented code and documenting each step to both solidify my own learning and create a portfolio-ready resource for others interested in sports analytics and machine learning.

By tackling this project, I not only practiced technical skills with Python and scikit-learn, but also developed a deeper appreciation for the iterative, analytical mindset required to turn raw data into actionable predictions—skills that are fundamental for any aspiring data scientist or AI practitioner.


## Project Goal

**Goal:**  
Predict NBA players’ points per game (PTS) using their season statistics, showcasing a complete machine learning workflow for your GitHub portfolio.

## Step-by-Step with Commented Code

### 1. Data Loading and Initial Exploration

```python
import pandas as pd

# Load the NBA high scorers dataset from CSV
df = pd.read_csv('nba_high_scorers_outliers.csv')

# Display the first 5 rows to get a sense of the data
print(df.head())

# Show info about columns, types, and missing values
print(df.info())

# Show summary statistics for numeric features
print(df.describe())
```
*This step ensures you understand your data’s structure and content.*

### 2. Data Cleaning

```python
# Check for missing values in each column
print(df.isnull().sum())

# Drop rows where the target (PTS) is missing
df = df.dropna(subset=['PTS'])

# Fill other missing values with zero (simple strategy)
df = df.fillna(0)

# Remove duplicate rows based on Player and Year
df = df.drop_duplicates(subset=['Player', 'Year'])
```
*Here, we ensure the data is clean and ready for analysis.*

### 3. Feature Selection and Engineering

```python
# List columns to drop: identifiers and the target
drop_cols = ['Rk', 'Player', 'Tm', 'PTS']  # 'PTS' is our target

# Separate features (X) and target (y)
X = df.drop(columns=drop_cols)
y = df['PTS']

# Convert categorical 'Pos' (position) into dummy/indicator variables
X = pd.get_dummies(X, columns=['Pos'], drop_first=True)
```
*We prepare features for modeling and encode categorical variables.*

### 4. Train/Test Split

```python
from sklearn.model_selection import train_test_split

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
*This ensures we can evaluate our model on unseen data.*

### 5. Exploratory Data Analysis (EDA)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot a correlation heatmap for all numeric features
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Plot the distribution of the target variable (PTS)
plt.figure()
sns.histplot(y, bins=30, kde=True)
plt.title('Distribution of Points Per Game')
plt.xlabel('PTS')
plt.show()
```
*Visualizations help us understand relationships and data distribution.*

### 6. Model Selection and Training

```python
from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf.fit(X_train, y_train)
```
*Random Forests are robust for tabular regression tasks.*

### 7. Model Evaluation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")    # Mean Absolute Error
print(f"RMSE: {rmse:.2f}")  # Root Mean Squared Error
print(f"R^2 Score: {r2:.2f}")  # R-squared score
```
*We assess the model’s accuracy using standard regression metrics.*

### 8. Feature Importance

```python
import numpy as np

# Get feature importances from the trained model
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

# Plot feature importances
plt.figure(figsize=(12,6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.tight_layout()
plt.show()
```
*This shows which features most influence the model’s predictions.*

### 9. Save the Model

```python
import joblib

# Save the trained model to a file for future use
joblib.dump(rf, 'nba_pts_predictor_rf.pkl')
```
*Saving the model enables easy reuse or deployment.*

## Full Code Block (Copy-Paste Ready)

```python
# NBA High Scorers Points Prediction Project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

# 1. Data Loading and Initial Exploration
df = pd.read_csv('nba_high_scorers_outliers.csv')
print(df.head())            # Show first 5 rows
print(df.info())            # Show data types and missing values
print(df.describe())        # Show summary statistics

# 2. Data Cleaning
print(df.isnull().sum())    # Check for missing values
df = df.dropna(subset=['PTS'])  # Drop rows with missing target
df = df.fillna(0)               # Fill other missing values with 0
df = df.drop_duplicates(subset=['Player', 'Year'])  # Remove duplicates

# 3. Feature Selection and Engineering
drop_cols = ['Rk', 'Player', 'Tm', 'PTS']  # Columns to drop
X = df.drop(columns=drop_cols)              # Features
y = df['PTS']                               # Target
X = pd.get_dummies(X, columns=['Pos'], drop_first=True)  # Encode 'Pos'

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Exploratory Data Analysis (EDA)
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

plt.figure()
sns.histplot(y, bins=30, kde=True)
plt.title('Distribution of Points Per Game')
plt.xlabel('PTS')
plt.show()

# 6. Model Selection and Training
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)  # Train the model

# 7. Model Evaluation
y_pred = rf.predict(X_test)  # Predict on test set
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# 8. Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns
plt.figure(figsize=(12,6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.tight_layout()
plt.show()

# 9. Save the Model
joblib.dump(rf, 'nba_pts_predictor_rf.pkl')
```

## Educational Summary

- **Data Loading:** Understand your data’s structure.
- **Cleaning:** Handle missing values and duplicates.
- **Feature Engineering:** Prepare features for modeling.
- **EDA:** Visualize relationships and distributions.
- **Modeling:** Train a Random Forest for regression.
- **Evaluation:** Assess performance with MAE, RMSE, and R².
- **Interpretation:** Visualize feature importance.
- **Deployment:** Save your trained model for future use.

