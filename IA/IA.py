# SEHS4696 Machine Learning for Data Mining
# Individual Assignment Template v24
#
# Name & Student ID:
#
# Apply Logistic Regression on Breast Cancer Diagnostics Prediction
#

# Enter the code to import the necessary package
# Place all your import statements here
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ** 1. Load and Understand the Data
#
# Enter the code to load the dataset into a dataframe called myDF
# Add other lines below to understand the dataset better

myDF = pd.read_csv("IA breast cancer v2d.csv")
print("The first 5 rows of the dataset:")
print(myDF.head())
print("The unique values of the diagnosis column:")
print(myDF["diagnosis"].value_counts())

# ** 2. Data Preprocessing
# -- 2.1 Drop Unnecessary Features
myDF.drop("id", axis=1, inplace=True)
print("The first 5 rows of the dataset after dropping the id column:")
print(myDF.head())
# -- 2.2 Encode strings into numbers
myDF["diagnosis"] = myDF["diagnosis"].map({"M": 1, "B": 0})


# -- 2.3 Handle missing data, if any
#    Look for any missing data using isnull().
#    Hints: 1. Refer to Data Preparation/Preprocessing of Lesson 4
#           2. Refer to slides about SimpleImputer in in Lesson 2


def impute_missing_data(data: pd.DataFrame):
    imputer = SimpleImputer(strategy="mean")
    data = imputer.fit_transform(data)
    return data


missing_data_series = myDF.isnull().sum()

if missing_data_series.sum() > 0:
    print("There are missing data")

    # Find the columns with missing data from the series
    missing_data_columns = missing_data_series[missing_data_series > 0].index.tolist()
    print("The columns with missing data are:")
    print(missing_data_columns)

    # Impute the missing data
    myDF[missing_data_columns] = impute_missing_data(myDF[missing_data_columns])

    # Check if there is still missing data
    print(
        f"The missing data after imputation: \
        {True if myDF.isnull().sum().sum() > 0 else False}"
    )
else:
    print("There is no missing data")

# -- 2.2 Feature Scaling
#    Use two scaling methods to produce two sets of data for the next steps
#    You should split the data into training and testing sets before scaling them


# Define features and target label
X = myDF.drop("diagnosis", axis=1)
y = myDF["diagnosis"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100
)

# Initialize two scalers
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

# Scale data using StandardScaler
X_train_standard = scaler_standard.fit_transform(X_train)
X_test_standard = scaler_standard.transform(X_test)

# Scale data using MinMaxScaler
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

# ** 3. Logistic Regression
# -- 3.1 Build and train the Model using the sets of data from above

# Initialize Logistic Regression model
model = LogisticRegression()

# Train models on the two sets of data
# Using StandardScaler data
model_standard = model.fit(X_train_standard, y_train)
yhat_standard = model_standard.predict(X_test_standard)

# Using MinMaxScaler data
model_minmax = model.fit(X_train_minmax, y_train)
yhat_minmax = model_minmax.predict(X_test_minmax)

# -- 3.2 Create and view the Confusion Matrix

# Generate confusion matrices for both models
cm_standard = confusion_matrix(y_test, yhat_standard)
cm_minmax = confusion_matrix(y_test, yhat_minmax)

print("Confusion matrix for the model built using StandardScaler data:")
print(cm_standard)
print("Confusion matrix for the model built using MinMaxScaler data:")
print(cm_minmax)

# Calculate accuracy, precision, recall, f1-score
# Accuracy
accuracy_standard = accuracy_score(y_test, yhat_standard)
accuracy_minmax = accuracy_score(y_test, yhat_minmax)

# Precision
precision_standard = precision_score(y_test, yhat_standard)
precision_minmax = precision_score(y_test, yhat_minmax)

# Recall
recall_standard = recall_score(y_test, yhat_standard)
recall_minmax = recall_score(y_test, yhat_minmax)

# F1-score
f1_standard = f1_score(y_test, yhat_standard)
f1_minmax = f1_score(y_test, yhat_minmax)

# Group the scores into a dataframe
scores = pd.DataFrame(
    {
        "Standard Scaler": [
            accuracy_standard,
            precision_standard,
            recall_standard,
            f1_standard,
        ],
        "MinMax Scaler": [
            accuracy_minmax,
            precision_minmax,
            recall_minmax,
            f1_minmax,
        ],
    },
    index=["Accuracy", "Precision", "Recall", "F1 Score"],
)
print(
    "The metrics of the models built using StandardScaler and MinMaxScaler are as follows:"
)
print(scores)

print(f"The difference in the accuracy is {accuracy_standard - accuracy_minmax}")

# ** 4. Summary
# -- Write a short summary of your findings in points form.
#    1. The accuracy of the model built using normalized data is ... than the model built using ...
#    2. The difference in the accuracy is .....
#    3. In this case, it is preferred to use the model built using .... for predicting unseen cases.

# The metrics of the models built using StandardScaler and MinMaxScaler are as follows:
#            Standard Scaler  MinMax Scaler
# Accuracy          0.964912       0.959064
# Precision         1.000000       1.000000
# Recall            0.913043       0.898551
# F1 Score          0.954545       0.946565
# 1.  The accuracy of the model built using standardize data is higher than the model built using MinMaxScaler data.
# 2.  The difference in the accuracy is 0.00584795
# 3.  In this case, it is preferred to use the model built using standardize data for predicting unseen cases.

# ** End of Assignment
