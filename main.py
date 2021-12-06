
#  _____ _ _              _       ______             _ _      _   _                 
# |_   _(_) |            (_)      | ___ \           | (_)    | | (_)                
#   | |  _| |_ __ _ _ __  _  ___  | |_/ / __ ___  __| |_  ___| |_ _  ___  _ __  ___ 
#   | | | | __/ _` | '_ \| |/ __| |  __/ '__/ _ \/ _` | |/ __| __| |/ _ \| '_ \/ __|
#   | | | | || (_| | | | | | (__  | |  | | |  __/ (_| | | (__| |_| | (_) | | | \__ \
#   \_/ |_|\__\__,_|_| |_|_|\___| \_|  |_|  \___|\__,_|_|\___|\__|_|\___/|_| |_|___/
                                                                                  
# AUTHOTS: Austin Lee, George Melek, Gianna Galard
# CSC412 PROFESSOR IMBERMAN
# DATE: 11/23/2021

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

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

# generate heatmap of train dataframe 
sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap='viridis') # change color whenever

# remove missing values from train dataframe
train_df = train_df.drop(['Cabin'], axis=1)

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
train_df['Embarked'] = train_df['Embarked'].fillna('S')
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
# print the boolean variable for each of the embarkment points
print(train_df['Embarked'].value_counts())

# Create a boolean for is male. 
#not sure if this is how you do it

train_df_sex = pd.get_dummies(train_df['Sex'])
isMale = train_df_sex['male']
isMale

