
#  _____ _ _              _       ______             _ _      _   _                 
# |_   _(_) |            (_)      | ___ \           | (_)    | | (_)                
#   | |  _| |_ __ _ _ __  _  ___  | |_/ / __ ___  __| |_  ___| |_ _  ___  _ __  ___ 
#   | | | | __/ _` | '_ \| |/ __| |  __/ '__/ _ \/ _` | |/ __| __| |/ _ \| '_ \/ __|
#   | | | | || (_| | | | | | (__  | |  | | |  __/ (_| | | (__| |_| | (_) | | | \__ \
#   \_/ |_|\__\__,_|_| |_|_|\___| \_|  |_|  \___|\__,_|_|\___|\__|_|\___/|_| |_|___/
                                                                                  
# AUTHORS: Gianna Galard, George Melek, Austin Li
# CSC412 PROFESSOR IMBERMAN
# DATE: 11/23/2021

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns 
# divide data into train and test sets
from sklearn.model_selection import train_test_split 
# logistic regression 
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# ---------------------- OUTPUT DATA

# print shape of train and test dataframes
print(train_df.shape) 
print(test_df.shape) 

# print first 5 rows of train dataframe
print(train_df.head())

# print datatype of train dataframe
print(train_df.dtypes)

#print train dataframe info
print(train_df.info())

# remove missing values from train dataframe
train_df = train_df.drop(['Cabin'], axis = 1)

# print the values removed from train dataframe
print(train_df.isnull().sum())

# select all columns that are objects
categorical_columns = train_df.select_dtypes(include=['object'])
# print number of categorical columns
print(f'There are {len(categorical_columns.columns.tolist())} categorical columns in the dataset:')
# for each column in categorical columns, print column name and number of unique values
for cols in categorical_columns.columns: 
    print(len(categorical_columns[cols].unique()),'labels in', cols)


# ---------------------- DATA CLEANING

# create boolean variable for each of the embarkment points
# for each value in the Embarked column
for uwu in train_df.Embarked.unique(): 
    # if it is a string (just to be safe lol)
    if type(uwu) == str: 
        # create a new column with the boolean value
        train_df['emb' + uwu] = (train_df.Embarked == uwu) * 1 

# create boolean variable for is male
train_df['is_male'] = (train_df.Sex == 'male') * 1 

# create boolean variable for has cabin
train_df.loc[:, 'has_cabin'] = 0
train_df.loc[train_df.Cabin.isna(), 'has_cabin'] = 1

# fill in missing age values
# replace with 100
train_df.loc[train_df.Age.isna(), 'Age'] = 100 

# split data in train and test sets
selected = train_df[["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]] 
Y = train_df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 7)

# one hot encoding 
one_hot_encoded_training_predictors = pd.get_dummies(selected) 
one_hot_encoded_training_predictors.head()
X = one_hot_encoded_training_predictors

# ---------------------- PICK 3 CLASSIFICTION ALGORITHMS
# 1. logistic regression
log = LogisticRegression(max_iter = 1000) 
log.fit(X_train, y_train)
y_pred_log = log.predict(X_test)
y_pred_log = log.predict(X_test)
print("logistic reg accuracy is: {:}" .format(log.score(X_test, y_test))) 
scores_accuracy = cross_val_score(log, X, Y, cv = 10, scoring = 'accuracy')
print('Cross Validation results:')
print(" logistic reg average accuracy is %2.3f" % scores_accuracy.mean())

# 2. decision tree classifier algorithm

# 3. catboost classifier algorithm