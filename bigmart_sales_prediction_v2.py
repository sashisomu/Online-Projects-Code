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
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import classification_report,confusion_matrix

os.chdir('C:/Users/Data_Science_with_Python/Bigmart Sales Prediction')
print(os.getcwd())

print(os.listdir())

#os.rename('train_v9rqX0R.csv','train.csv')
#os.rename('test_AbJTz2l.csv','test.csv')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

shutil.copyfile('train.csv','train_backup.csv')
shutil.copyfile('test.csv','test_backup.csv')

#print(train.info())
print(train['Item_Identifier'].value_counts())
print(train['Item_Fat_Content'].value_counts())
print(train['Item_Type'].value_counts())
print(train['Outlet_Identifier'].value_counts())
print(train['Outlet_Size'].value_counts())
print(train['Outlet_Location_Type'].value_counts())
print(train['Outlet_Type'].value_counts())


labels = ['Item_Identifier','Item_Fat_Content','Item_Type',
          'Outlet_Size','Outlet_Location_Type','Outlet_Type']

cat_label = lambda x:x.astype('category')
train[labels] = train[labels].apply(cat_label,axis=0)

float_type_cols = ['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year',
                   'Item_Outlet_Sales']

#print(train.info())

#Exploratory data analysis

# Univariate Anaysis of contineous variables

for col in float_type_cols:
    
    plt.hist(train[col],30)
    plt.xlabel(col)
    plt.show()

# Univariate Anaysis of categorical variables
    plt.clf()

# Fill the missing values

#print(train['Item_Weight'].isnull().sum())

train['Item_Weight'] = train['Item_Weight'].fillna(train['Item_Weight'].mean())

#print(train['Outlet_Size'].isnull().sum())

train['Outlet_Size'] = train['Outlet_Size'].fillna(train['Outlet_Size'].value_counts().index[0])
#print(train.info())
# Convert categorical variables into numeric values

train['Item_Fat_Content'].replace(('Low Fat','Regular','LF','reg','low fat'),(1,2,1,2,1),inplace = True)
train['Item_Fat_Content'].astype('int')

train['Item_Type'] = train['Item_Type'].cat.codes
train['Item_Type'] = train['Item_Type'] + 1
train['Item_Type'].astype('int')

train['Outlet_Size'] = train['Outlet_Size'].cat.codes
train['Outlet_Size'] = train['Outlet_Size'] + 1
train['Outlet_Size'].astype('int')

train['Outlet_Location_Type'] = train['Outlet_Location_Type'].cat.codes
train['Outlet_Location_Type'] = train['Outlet_Location_Type'] + 1
train['Outlet_Location_Type'].astype('int')

train['Outlet_Type'] = train['Outlet_Type'].cat.codes
train['Outlet_Type'] = train['Outlet_Type'] + 1
train['Outlet_Type'].astype('int')

print(train.info())
#Bi-Variate Analysis


for col in train.columns.values:
    if col != 'Item_Outlet_Sales':        
        plt.scatter(train[col], train['Item_Outlet_Sales'])
        plt.xlabel(col)
        plt.show()
        
#train.to_csv('a.csv')
# Fitting the Lasso regression model to know the key parameters 
col_to_drop = ['Item_Identifier','Item_Outlet_Sales','Outlet_Identifier']
#X = train.drop(col_to_drop, axis = 1).values
X = train['Item_MRP'].values
X = X.reshape(-1,1)
y = train['Item_Outlet_Sales'].values

reg = LinearRegression(normalize=True)
reg.fit(X,y)
reg_coef = reg.fit(X,y).coef_

print(reg_coef)
# Plot coefficients
#plt.plot(range(len(df_columns)),lasso_coef)
#plt.xticks(range(len(train_columns)),train_columns.values,rotation=60)
#plt.margins(0.02)
#plt.show()

# Preparation of test set for predictions

test[labels] = test[labels].apply(cat_label,axis=0)
test['Item_Weight'] = test['Item_Weight'].fillna(test['Item_Weight'].mean())

for col in labels:
    if col != 'Item_Identifier':
        test[col] = test[col].cat.codes
        test[col] = test[col] + 1
        test[col].astype('int')

print(test.info()) 
#X_test = test.drop(['Item_Identifier','Outlet_Identifier'], axis = 1).values
X_test = test['Item_MRP'].values
X_test = X_test.reshape(-1,1)
predictions = reg.predict(X_test)
df = test[['Item_Identifier','Outlet_Identifier']]
predictions = pd.DataFrame(predictions,columns=['Item_Outlet_Sales'])
plt.hist(predictions['Item_Outlet_Sales'],30)
plt.show()
predictions = pd.concat([df,predictions],axis=1)
predictions.to_csv('submission.csv', index = False)
#df.to_csv('submission1.csv')









