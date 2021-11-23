
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

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# print the first 5 rows of the dataset
print(train_df.head())

# determine how many rows contain missing values
print(train_df.isnull().sum())

# print the unique values for the columns "embarked", "cabin", and "ticket"
print(train_df.embarked.unique())
print(train_df.cabin.unique())
print(train_df.ticket.unique())