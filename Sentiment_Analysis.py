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

os.chdir('C:/Users/Data_Science_with_Python/Sentiment_Analysis')
print(os.getcwd())

print(os.listdir())

#os.rename('train_2kmZucJ.csv','train.csv')
#os.rename('test_oJQbWVk.csv','test.csv')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

shutil.copyfile('train.csv','train_backup.csv')
shutil.copyfile('test.csv','test_backup.csv')

print(train.columns)
y = train.label
X = train.tweet

#y_test = test.label
#X_test = test.tweet

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
print(metrics.f1_score(y_test,y_pred))
#Fit the model using whole train and test set now

y = train.label
X = train.tweet

count_train_new = count_vectorizer.fit_transform(X)
nb_classifier_new = MultinomialNB()

nb_classifier_new.fit(count_train_new,y)

#Make the predictions on true train set

X_test_new = test.tweet
count_test_new = count_vectorizer.transform(X_test_new)

y_test_new = nb_classifier_new.predict(count_test_new)

y_test_new = pd.DataFrame(y_test_new,columns=['label'])

df = test.id
df = pd.concat([df,y_test_new],axis = 1)
df.to_csv('submission.csv',index=False)


