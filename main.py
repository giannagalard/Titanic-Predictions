
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
from sklearn import tree
# svm
from sklearn.svm import SVC
# accuracy 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, plot_confusion_matrix



# import data from csv file for training
dataset = pd.read_csv('train.csv')
# dataset = pd.read_csv('/Users/george/Downloads/train.csv')
dataset # print dataset

# shape of train dataframes
dataset.shape 

# first 5 rows of train dataframe
dataset.head()

# print train dataframe info 
dataset.info()

# generate statistics
dataset.describe()

# output columns with labels
# select all columns that are objects
categorical_columns = dataset.select_dtypes(include=['object'])
# print number of categorical columns
print(f'There are {len(categorical_columns.columns.tolist())} categorical columns in the dataset:')
# for each column in categorical columns, print column name and number of unique values
for cols in categorical_columns.columns: 
    print(cols,':', len(categorical_columns[cols].unique()),'labels')


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

# split data in train and test sets
selected = dataset[["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]] 
Y = dataset["Survived"]

# drop rows with nul and nan
dataset.dropna(how = 'all')

# Split data and apply label encoder to the sex and embarked columns
featName = ["Pclass", "Age", "Sex", "SibSp", "Parch", "Embarked"]
X = dataset[featName]
X # print

label_Encoder = LabelEncoder()
X["Sex"] = label_Encoder.fit_transform(X["Sex"])
X["Embarked"] = label_Encoder.fit_transform(X["Embarked"])
X # print
# i get a scary warning but ... it works ??? so not worried lol

y = dataset["Survived"]
y # print

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)

# create an accurate model
# 1. Logistic Regression Algorithm
# 2. Decision Tree Algorithm
# 3. Support Vector Machine Algorithm

                                                                          
#  __            _     _   _        _____                         _         
# |  |   ___ ___|_|___| |_|_|___   | __  |___ ___ ___ ___ ___ ___|_|___ ___ 
# |  |__| . | . | |_ -|  _| |  _|  |    -| -_| . |  _| -_|_ -|_ -| | . |   |
# |_____|___|_  |_|___|_| |_|___|  |__|__|___|_  |_| |___|___|___|_|___|_|_|
#           |___|                            |___|                          

logitModel = LogisticRegression().fit(X_train, y_train) # create logistic regression model
y_pred = logitModel.predict(X_test) # predict test set

accuracy = accuracy_score(y_test, y_pred) # calculate accuracy

# print accuracy
print("Accuracy Score -> {}".format(accuracy))

y_pred_LR = logitModel.predict(X_test)
print(classification_report(y_test, y_pred_LR))
print("ROC AUC Score is {}".format(roc_auc_score(y_test, y_pred_LR)))

scores_accuracy = cross_val_score(logitModel, X, y, cv = 9, scoring = 'accuracy')
print('Cross Validation results:')
print("Logistic reg average accuracy is %2.3f" % scores_accuracy.mean())

print("Confusion Matrix for Logistic Regression")
displr = plot_confusion_matrix(logitModel, X_test, y_test,cmap=plt.cm.Blues, values_format='d')

#  ____          _     _            _____            
# |    \ ___ ___|_|___|_|___ ___   |_   _|__ ___ ___ 
# |  |  | -_|  _| |_ -| | . |   |    | ||  _| -_| -_|
# |____/|___|___|_|___|_|___|_|_|    |_||_| |___|___|                                             

dta = DecisionTreeClassifier(max_depth = 5).fit(X_train, y_train) # create decision tree model
y_pred = dta.predict(X_test) # predict test set

accuracy = accuracy_score(y_test, y_pred) # calculate accuracy

# print accuracy
print("Accuracy Score -> {}".format(accuracy))

print(classification_report(y_pred,y_test))

print("Confusion Matrix for Decision Tree Classifier")
decision_confusion = plot_confusion_matrix(dta, X_test, y_test ,cmap=plt.cm.Blues , values_format='d')

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dta,
                   feature_names = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
       'Cabin', 'male', 'C', 'Q', 'S'],
                   max_depth = 3,
                   filled=True)

text_representation = tree.export_text(dta)
print(text_representation)

#  _____ _____ _____ 
# |   __|  |  |     |
# |__   |  |  | | | |
# |_____|\___/|_|_|_|

svcModel = SVC(kernel = 'linear').fit(X_train, y_train) # create svm model
y_pred = svcModel.predict(X_test) # predict test set

accuracy = accuracy_score(y_test, y_pred) # calculate accuracy

# print accuracy
print("Accuracy Score -> {}".format(accuracy))

print(classification_report(y_test, y_pred))

print("Confusion Matrix for Support Vector Machines")
support_confusion = plot_confusion_matrix(svcModel, X_test, y_test ,cmap=plt.cm.Blues , values_format='d')

