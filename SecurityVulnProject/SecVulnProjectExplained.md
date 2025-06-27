# Simple AI/ML Learning Project: Vulnerability Severity Classification

This step-by-step project uses the provided "Security-Vulnerabilities.csv" dataset to build a machine learning model that predicts the severity of a software vulnerability based on its summary text. The goal is to showcase your AI/ML skills in data preprocessing, feature engineering, model training, and evaluation—all with clear code explanations and educational summaries after each step.

## Step 1: Load and Explore the Data

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('Security-Vulnerabilities.csv')

# Show the first few rows
print(df.head())

# Show basic info about the dataset
print(df.info())
```
**Comments:**
- Import `pandas` for data handling.
- Read the CSV file into a DataFrame.
- Display the first few rows to understand the data structure.
- Print info to check for missing values and data types.

**Summary:**  
We loaded the vulnerability dataset, which includes columns like Title, Date, Severity, Summary, and Link. Our target variable for classification will be the "Severity" column.

## Step 2: Data Preprocessing

```python
# Drop rows with missing values in 'Summary' or 'Severity'
df = df.dropna(subset=['Summary', 'Severity'])

# For simplicity, keep only the columns we need
df = df[['Summary', 'Severity']]

# Check unique values in Severity
print(df['Severity'].value_counts())
```
**Comments:**
- Remove rows missing crucial information.
- Focus on the summary text and severity label.
- Check the distribution of severity classes.

**Summary:**  
We cleaned the data by removing incomplete rows and focusing on the summary and severity. The severity classes (e.g., 'Low', 'Moderate', 'High', 'Critical') are our prediction targets.

## Step 3: Feature Engineering (Text Vectorization)

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['Summary'], df['Severity'], test_size=0.2, random_state=42, stratify=df['Severity'])

# Convert summary text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=1000)  # Limit to 1000 features for simplicity
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```
**Comments:**
- Split data into training and testing sets, stratifying by severity to keep class balance.
- Use TF-IDF vectorization to convert text summaries into numerical features for the ML model.

**Summary:**  
We transformed the summary text into TF-IDF vectors, which capture the importance of words and phrases in each summary for use in machine learning.

## Step 4: Model Training

```python
from sklearn.linear_model import LogisticRegression

# Train a simple logistic regression classifier
clf = LogisticRegression(max_iter=200)
clf.fit(X_train_tfidf, y_train)
```
**Comments:**
- Import and initialize a logistic regression classifier.
- Train the model on the TF-IDF features and severity labels.

**Summary:**  
We trained a logistic regression model, a common baseline for text classification tasks, to predict vulnerability severity from summary text.

## Step 5: Model Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix

# Predict on the test set
y_pred = clf.predict(X_test_tfidf)

# Show classification metrics
print(classification_report(y_test, y_pred))

# Show confusion matrix
print(confusion_matrix(y_test, y_pred))
```
**Comments:**
- Make predictions on the test set.
- Print a classification report (precision, recall, F1-score) and confusion matrix to evaluate performance.

**Summary:**  
We evaluated the model's performance. The classification report shows how well the model predicts each severity class, and the confusion matrix helps identify common misclassifications.

## Step 6: Showcase and Interpret Results

```python
# Show a few predictions with their summaries
for i in range(5):
    print(f"Summary: {X_test.iloc[i]}")
    print(f"Actual Severity: {y_test.iloc[i]}, Predicted: {y_pred[i]}\n")
```
**Comments:**
- Display a few example summaries with their actual and predicted severity to interpret the model's predictions.

**Summary:**  
By reviewing individual predictions, we can better understand the model's strengths and weaknesses, and explain results to stakeholders.

# Project Recap

- **Goal:** Predict vulnerability severity from summary text using ML.
- **Process:** Data loading, cleaning, text vectorization, model training, and evaluation.
- **Skills Demonstrated:** Data preprocessing, feature engineering, text classification, model evaluation, and interpretability.
- **Extensions:** You could try more advanced models (e.g., SVM, Random Forest, or BERT), hyperparameter tuning, or more detailed text preprocessing for further improvement.

This project is a concise, practical showcase of your AI/ML workflow and skills using a real-world security dataset[1].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/5013513/6120a161-8891-4292-99da-8ee7d0433848/Security-Vulnerabilities.csv
