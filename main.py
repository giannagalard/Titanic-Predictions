
#  _____ _ _              _       ______             _ _      _   _                 
# |_   _(_) |            (_)      | ___ \           | (_)    | | (_)                
#   | |  _| |_ __ _ _ __  _  ___  | |_/ / __ ___  __| |_  ___| |_ _  ___  _ __  ___ 
#   | | | | __/ _` | '_ \| |/ __| |  __/ '__/ _ \/ _` | |/ __| __| |/ _ \| '_ \/ __|
#   | | | | || (_| | | | | | (__  | |  | | |  __/ (_| | | (__| |_| | (_) | | | \__ \
#   \_/ |_|\__\__,_|_| |_|_|\___| \_|  |_|  \___|\__,_|_|\___|\__|_|\___/|_| |_|___/
                                                                                  
# AUTHORS: Gianna Galard, George Melek, Austin Li
# CSC412 PROFESSOR IMBERMAN
# DATE: 11/23/2021

# import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# split data
from sklearn.model_selection import train_test_split
# encoding
from sklearn.preprocessing import LabelEncoder
# logistic regression
from sklearn.linear_model import LogisticRegression
# decision tree
from sklearn.tree import DecisionTreeClassifier
# svm
from sklearn.svm import SVC
# accuracy 
from sklearn.metrics import accuracy_score

# import data from csv file for training
dataset = pd.read_csv('train.csv')
dataset # print dataset

# divide data into train and test sets
from sklearn.model_selection import train_test_split
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# look at the data and see if there are any summary statistics that might give you some insights

# print shape of train and test sets
print(X_train.shape)
print(X_test.shape)

# print first 5 rows of train dataset
print(dataset.head())

# print train dataframe info
dataset.info()

# output columns with labels
# select all columns that are objects
categorical_columns = dataset.select_dtypes(include=['object'])
# print number of categorical columns
print(f'There are {len(categorical_columns.columns.tolist())} categorical columns in the dataset:')
# for each column in categorical columns, print column name and number of unique values
for cols in categorical_columns.columns: 
    print(cols,':', len(categorical_columns[cols].unique()),'labels')

# Data cleaning and feature engineering

# create boolean for each of the embarkment points
# for each value in the Embarked column
# instead of value use uwu for funsies ahaha
for uwu in dataset.Embarked.unique(): 
    # if it is a string (just to be safe lol)
    if type(uwu) == str: 
        # create a new column with the boolean value
        dataset['Embark' + uwu] = (dataset.Embarked == uwu) * 1 

# create boolean for is male
dataset['isMale'] = (dataset.Sex == 'male') * 1 

# create boolean for has cabin
dataset.loc[:, 'has_cabin'] = 0
dataset.loc[dataset.Cabin.isna(), 'has_cabin'] = 1

# check for missing values in the data
print(dataset.isnull().sum())

# fill missing age values as 100 
dataset['Age'] = dataset['Age'].fillna(100)
dataset

# Split data and apply label encoder to the sex and embarked columns
featName = ["Pclass", "Age", "Sex", "SibSp", "Parch", "Embarked"]
X = dataset[featName]
X # print

# drop rows with nul and nan
dataset.dropna(how = 'all')


label_Encoder = LabelEncoder()
X["Sex"] = label_Encoder.fit_transform(X["Sex"])
X["Embarked"] = label_Encoder.fit_transform(X["Embarked"])
X # print
# i get a scary warning but ... it works ??? so not worried lol

y = dataset["Survived"]
y # print


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)

# create an accurate model
# 1. Logistic Regression Algorithm
# 2. Decision Tree Algorithm
# 3. Support Vector Machine Algorithm

#    _                                                           
#   //             _/_                                           
#  // __ _,  o _   /  o _.   __  _  _,  __  _  _   _   o __ ____ 
# </_(_)(_)_<_/_)_<__<_(__  / (_</_(_)_/ (_</_/_)_/_)_<_(_)/ / <_
#        /|                         /|                           
#       |/                         |/                            

logitModel = LogisticRegression().fit(X_train, y_train) # create logistic regression model
y_pred = logitModel.predict(X_test) # predict test set

accuracy = accuracy_score(y_test, y_pred) # calculate accuracy

#   __/   _   __   .  ,    .  _,_ ,__,     -/- ,_   _   _ 
# _(_/(__(/__(_,__/__/_)__/__(_/_/ / (_   _/__/ (__(/__(/_

dta = DecisionTreeClassifier(max_depth = 5).fit(X_train, y_train) # create decision tree model
y_pred = dta.predict(X_test) # predict test set

accuracy = accuracy_score(y_test, y_pred) # calculate accuracy

#   _     _  
# _)  \/ //) 

svcModel = SVC(kernel = 'linear').fit(X_train, y_train) # create svm model
y_pred = svcModel.predict(X_test) # predict test set

accuracy = accuracy_score(y_test, y_pred) # calculate accuracy