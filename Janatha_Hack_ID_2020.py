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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

os.chdir('C:/Users/Hakathons/Janatha_Hack_ID_2020_ML')
print(os.getcwd())
print(os.listdir())

#os.rename('train_2kmZucJ.csv','train.csv')
#os.rename('test_oJQbWVk.csv','test.csv')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#shutil.copyfile('train.csv','train_backup.csv')
#shutil.copyfile('test.csv','test_backup.csv')

print(train.columns)

X = train.TITLE.str.cat(train.ABSTRACT,' ').values
train['Physics'].replace((1,0),(100,0), inplace = True)
train['Mathematics'].replace((1,0),(200,0), inplace = True)
train['Statistics'].replace((1,0),(300,0), inplace = True)
train['Quantitative Biology'].replace((1,0),(400,0), inplace = True)
train['Quantitative Finance'].replace((1,0),(500,0), inplace = True)

y = train['Computer Science']+train['Physics']+train['Mathematics']+train['Statistics']+train['Quantitative Biology']+train['Quantitative Finance']

#train['cat'] = train[train.columns[3:]].apply(lambda x: ','.join(x.dropna().astype(int).astype(str)),axis=1)
train = pd.concat([train,y],axis = 1)
train.to_csv('a.csv')

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 53)

count_vectorizer = CountVectorizer(stop_words='english')

count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

print(count_test)
print(y_train)

nb_classifier = MultinomialNB()
nb_classifier.fit(count_train,y_train)

y_pred = nb_classifier.predict(count_test)

score = metrics.accuracy_score(y_test, y_pred) 
print(score)

cm = metrics.confusion_matrix(y_test, y_pred) 
print(cm) 
print(metrics.f1_score(y_test,y_pred,average = 'micro'))

#Fit the model using whole train and test set now

count_train_new = count_vectorizer.fit_transform(X)

nb_classifier_new = MultinomialNB()

nb_classifier_new.fit(count_train_new,y)

#Make the predictions on true train set

X_test_new = test.TITLE.str.cat(train.ABSTRACT,' ').values

count_test_new = count_vectorizer.transform(X_test_new)

y_test_new = nb_classifier_new.predict(count_test_new)
#y_test_new = y_test_new.reshape(8989,7)
y_test_new = pd.DataFrame(y_test_new,columns=['label'])
y_test_new_1 = pd.DataFrame(columns = ['Computer Science','Physics','Mathematics','Statistics','Quantitative Biology','Quantitative Finance'])

df = test.ID
df = pd.concat([df,y_test_new_1,y_test_new],axis = 1)
df = pd.get_dummies(df['label'],)
df1 = test.ID
df = pd.concat([df1,df],axis = 1)
df = df.rename(columns = {1:'Computer Science',100:'Physics',200:'Mathematics',300:'Statistics',400:'Quantitative Biology',500:'Quantitative Finance'})

df.loc[df[101]==1,"Computer Science"] = 1
df.loc[df[101]==1,"Physics"] = 1

df.loc[df[201]==1,"Computer Science"] = 1
df.loc[df[201]==1,"Mathematics"] = 1

df.loc[df[301]==1,"Computer Science"] = 1
df.loc[df[301]==1,"Statistics"] = 1

df.loc[df[401]==1,"Computer Science"] = 1
df.loc[df[401]==1,"Quantitative Biology"] = 1

df.loc[df[700]==1,"Quantitative Finance"] = 1
df.loc[df[700]==1,"Statistics"] = 1

df = df.drop([101,201,301,401,700],axis = 1)
  
df.to_csv('submission.csv',index=False)















