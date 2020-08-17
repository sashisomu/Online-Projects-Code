# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:44:24 2020

@author: somabhupal
"""

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

os.chdir('C:/Users/Hakathons/Customer_Segmentation')
path = os.getcwd()

#os.rename('Train_aBjfeNk.csv','train.csv')
#os.rename('Test_LqhgPWU.csv','test.csv')
print(os.listdir())

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.head())

print(train['Gender'].value_counts())
print(train['Ever_Married'].value_counts())
print(train['Graduated'].value_counts())
print(train['Profession'].value_counts())
print(train['Spending_Score'].value_counts())
print(train['Family_Size'].value_counts())
print(train['Var_1'].value_counts())
print(train['Segmentation'].value_counts())

print(train['Age'].describe())
print(train['Work_Experience'].describe())

train['Ever_Married'] = train['Ever_Married'].fillna(train['Ever_Married'].value_counts().index[0]) 
train['Work_Experience'] = train['Work_Experience'].fillna(train['Work_Experience'].mean()) 
train['Graduated'] = train['Graduated'].fillna(train['Graduated'].value_counts().index[0]) 
train['Profession'] = train['Profession'].fillna(train['Profession'].value_counts().index[0]) 
train['Family_Size'] = train['Family_Size'].fillna(train['Family_Size'].value_counts().index[0]) 
train['Var_1'] = train['Var_1'].fillna(train['Var_1'].value_counts().index[0]) 

print(train.isnull().sum())

for col in train.columns.values:

    if train[col].dtype == 'object':
        train[col] = train[col].astype('category')
        train[col] = train[col].cat.codes
        train[col] = train[col] + 1

print(train.info())

X = train.drop(['ID','Segmentation'],axis = 1).values
y = train['Segmentation'].values

y = y.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,
                                                    random_state = 43)

logreg = LogisticRegression()
logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

print(confusion_matrix(y_test,y_pred))

#Prepare the test set

test['Ever_Married'] = test['Ever_Married'].fillna(test['Ever_Married'].value_counts().index[0]) 
test['Work_Experience'] = test['Work_Experience'].fillna(test['Work_Experience'].mean()) 
test['Graduated'] = test['Graduated'].fillna(test['Graduated'].value_counts().index[0]) 
test['Profession'] = test['Profession'].fillna(test['Profession'].value_counts().index[0]) 
test['Family_Size'] = test['Family_Size'].fillna(test['Family_Size'].value_counts().index[0]) 
test['Var_1'] = test['Var_1'].fillna(test['Var_1'].value_counts().index[0]) 

for col in test.columns.values:
    if test[col].dtype == 'object':
        test[col] = test[col].astype('category')
        test[col] = test[col].cat.codes
        test[col] = test[col] + 1

df = test['ID']

X_test_new = test.drop(['ID'],axis = 1).values 
prediction = logreg.predict(X_test_new)
prediction = pd.DataFrame(prediction,columns = ['Segmentation'])
prediction.replace((1,2,3,4),('A','B','C','D'), inplace = True)
prediction = pd.concat([df,prediction],axis = 1)
prediction.to_csv('submission.csv',index = False)


