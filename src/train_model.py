import numpy as np
import pandas as pd
import os

from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, classification_report, recall_score, confusion_matrix,
    roc_auc_score, precision_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import OrdinalEncoder

from catboost import CatBoostClassifier, Pool
import joblib
from sklearn.preprocessing import LabelEncoder


data_path = "../data/churn_data.csv"
df = pd.read_csv(data_path)

# Convert TotalCharges to numeric, filling NaN values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)

# Convert SeniorCitizen to object
df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)

# Replace 'No phone service' and 'No internet service' with 'No' for certain columns
df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
columns_to_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for column in columns_to_replace:
    df[column] = df[column].replace('No internet service', 'No')

# Convert 'Churn' categorical variable to numeric
df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})


label_encoders = {}
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Create the StratifiedShuffleSplit object
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=64)

train_index, test_index = next(strat_split.split(df, df["Churn"]))

# Create train and test sets
strat_train_set = df.loc[train_index]
strat_test_set = df.loc[test_index]

X_train = strat_train_set.drop("Churn", axis=1)
y_train = strat_train_set["Churn"].copy()

X_test = strat_test_set.drop("Churn", axis=1)
y_test = strat_test_set["Churn"].copy()


# Initialize and fit CatBoostClassifier
cat_model = CatBoostClassifier(verbose=False, random_state=0, scale_pos_weight=3)
cat_model.fit(X_train, y_train, cat_features=categorical_columns, eval_set=(X_test, y_test))

# Predict on test set
y_pred = cat_model.predict(X_test)

# Calculate evaluation metrics
accuracy, recall, roc_auc, precision = [round(metric(y_test, y_pred), 4) for metric in [accuracy_score, recall_score, roc_auc_score, precision_score]]

# Create a DataFrame to store results
model_names = ['CatBoost_Model']
result = pd.DataFrame({'Accuracy': accuracy, 'Recall': recall, 'Roc_Auc': roc_auc, 'Precision': precision}, index=model_names)

# Print results
print(result)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Initialize and fit Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set using Random Forest
y_pred_rf = rf_model.predict(X_test)

# Calculate evaluation metrics for Random Forest
accuracy_rf = round(accuracy_score(y_test, y_pred_rf), 4)
recall_rf = round(recall_score(y_test, y_pred_rf), 4)
roc_auc_rf = round(roc_auc_score(y_test, y_pred_rf), 4)
precision_rf = round(precision_score(y_test, y_pred_rf), 4)

# Initialize and fit SVM Classifier
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Predict on test set using SVM
y_pred_svm = svm_model.predict(X_test)

# Calculate evaluation metrics for SVM
accuracy_svm = round(accuracy_score(y_test, y_pred_svm), 4)
recall_svm = round(recall_score(y_test, y_pred_svm), 4)
roc_auc_svm = round(roc_auc_score(y_test, y_pred_svm), 4)
precision_svm = round(precision_score(y_test, y_pred_svm), 4)

# Create a DataFrame to store results
results = pd.DataFrame({
    'Accuracy': [accuracy, accuracy_rf, accuracy_svm],
    'Recall': [recall, recall_rf, recall_svm],
    'Roc_Auc': [roc_auc, roc_auc_rf, roc_auc_svm],
    'Precision': [precision, precision_rf, precision_svm]
}, index=['CatBoost_Model', 'RandomForest_Model', 'SVM_Model'])

print(results)

# Save the model in the 'model' directory
model_dir = "../model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

cat_model_path = os.path.join(model_dir, "catboost_model.cbm")
cat_model.save_model(cat_model_path)

rf_model_path = os.path.join(model_dir, "random_forest_model.pkl")
joblib.dump(rf_model, rf_model_path)

svm_model_path = os.path.join(model_dir, "svm_model.pkl")
joblib.dump(svm_model, svm_model_path)

# Saving the train test data
joblib.dump(X_train, '../data/X_train.pkl')
joblib.dump(X_test, '../data/X_test.pkl')
joblib.dump(y_train, '../data/y_train.pkl')
joblib.dump(y_test, '../data/y_test.pkl')

parquet_path = '../data/churn_data_processed.parquet'
df.to_parquet(parquet_path)