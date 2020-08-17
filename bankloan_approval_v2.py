# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

os.chdir('C:/Users/Data_Science_with_Python/loan_prediction')
print(os.getcwd())
print(os.listdir())

#os.rename('train_ctrUa4K.csv','train.csv')
#os.rename('test_lAUu6dG.csv','test.csv')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#shutil.copyfile('train.csv','train_backup.csv')
#shutil.copyfile('test.csv','test_backup.csv')

#print(train.columns)
#print(test.columns)

labels = ['Loan_ID','Gender','Married','Dependents','Education','Self_Employed',
          'Property_Area','Loan_Status']

other_labels = ['ApplicantIncome','CoapplicantIncome','LoanAmount',
                'Loan_Amount_Term','Credit_History']

# Convert data types to category

cat_label = lambda x:x.astype('category')
train[labels] = train[labels].apply(cat_label, axis = 0)

# Fill NAs

for col in labels:
    train[col] = train[col].fillna(train[col].value_counts().index[0])
        
for col in other_labels:
    train[col] = train[col].fillna(train[col].mean())
    
# Convert Y data as numerical data

train['Loan_Status'] = train['Loan_Status'].cat.codes
train['Loan_Status'].astype('int')

# Convert X Data as numerical data

#train['Dependents'] = train['Dependents'].str.split('+')
#train['Dependents'].astype('int')

#print(train.info())
#for col in labels:
    #if col != 'Loan_ID' or col != 'Dependents':
    #    print(train[col].value_counts())
   #     train[col] = train[col].cat.codes
  #      train[col] = train[col] + 1
 #       train[col].astype('int')
#train[train['Dependents'] == '3+'] = 3

#train['Loan_Status'].replace(('Y','N'),(2,1),inplace=True)
train['Gender'].replace(('Male','Female'),(2,1),inplace=True)
train['Married'].replace(('Yes','No'),(2,1),inplace=True)
train['Education'].replace(('Graduate','Not Graduate'),(2,1),inplace=True)
train['Self_Employed'].replace(('Yes','No'),(2,1),inplace=True)
train['Property_Area'].replace(('Rural','Urban','Semiurban'),(1,3,2),inplace=True)
train['Dependents'].replace(('0','1','2','3+'),(1,2,3,4),inplace=True)

train.to_csv('a.csv',index = False)

# Exploratory Data ANalysis

#Univariate Analysis

#for col in labels:
 #   sns.countplot(x=col,data = train, hue = 'Loan_Status')
  #  plt.xlabel(col)
   # plt.show()

#for col in other_labels:
 #   plt.hist(train[col],25)
  #  plt.show()
    

#Fitting the model

X  = train.drop(['Loan_ID','Loan_Status'],axis = 1).values
y =  train['Loan_Status'].values


logreg = LogisticRegression()
logreg.fit(X,y)

# Shape the test set
labels_test = ['Loan_ID','Gender','Married','Dependents','Education','Self_Employed',
          'Property_Area']
test[labels_test] = test[labels_test].apply(cat_label, axis = 0)

# Fill NAs
for col in labels_test:
    test[col] = test[col].fillna(test[col].value_counts().index[0])
        
for col in other_labels:
    test[col] = test[col].fillna(test[col].mean())

#convert cat variables to numeric in test set
test['Gender'].replace(('Male','Female'),(2,1),inplace=True)
test['Married'].replace(('Yes','No'),(2,1),inplace=True)
test['Education'].replace(('Graduate','Not Graduate'),(2,1),inplace=True)
test['Self_Employed'].replace(('Yes','No'),(2,1),inplace=True)
test['Property_Area'].replace(('Rural','Urban','Semiurban'),(1,3,2),inplace=True)
test['Dependents'].replace(('0','1','2','3+'),(1,2,3,4),inplace=True)

df = pd.DataFrame(test['Loan_ID'])

X_test = test.drop(['Loan_ID'], axis = 1)
y_pred = logreg.predict(X_test)
predictions = pd.DataFrame(y_pred,columns = ['Loan_Status'])
predictions['Loan_Status'].replace((0,1),('N','Y'),inplace = True)
predictions = pd.concat([df,predictions],axis = 1)
predictions.to_csv('submission.csv', index = False)
#print(logreg.score(X_test,y_pred))


