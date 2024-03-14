# SEHS4696 Machine Learning for Data Mining 
# Individual Assignment Template v24
#
# Name & Student ID: CHAN Tai Man John 18123456s
#
# Apply Logistic Regression on Breast Cancer Diagnostics Prediction
#

# Enter the code to import the necessary package
# Place all your import statements here

# ** 1. Load and Understand the Data
#
# Enter the code to load the dataset into a dataframe called myDF
# Add other lines below to understand the dataset better
myDF['diagnosis'].value_counts()

# ** 2. Data Preprocessing
# -- 2.1 Drop Unnecessary Features
myDF.drop('id',axis=1,inplace=True)
# -- 2.2 Encode strings into numbers
myDF['diagnosis'] = myDF['diagnosis'].map({'M':1,'B':0})
# -- 2.3 Handle missing data, if any
#    Look for any missing data using isnull(). 
#    Hints: 1. Refer to Data Preparation/Preprocessing of Lesson 4
#           2. Refer to slides about SimpleImputer in in Lesson 2


# -- 2.2 Feature Scaling
#    Use two scaling methods to produce two sets of data for the next steps
#    You should split the data into training and testing sets before scaling them

# ** 3. Logistic Regression
# -- 3.1 Build and train the Model using the sets of data from above
# -- 3.2 Create and view the Confusion Matrix

# ** 4. Summary
# -- Write a short summary of your findings in points form.
#    1. The accuracy of the model built using normalized data is ... than the model built using ...
#    2. The difference in the accuracy is .....
#    3. In this case, it is preferred to use the model built using .... for predicting unseen cases.

# ** End of Assignment

