#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:46:26 2019

@author: denkal
"""
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
train_df = pd.read_csv("data/num_train.csv")
test_df = pd.read_csv("data/num_test.csv")

#Making the training dataset for the model
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]


def split_values(x,y): return x[:y], x[y:]
count = 60
number = len(X_train) - count 
x_train, x_val = split_values(X_train,number)
y_train, y_val = split_values(Y_train,number)
x_train.shape,y_train.shape,x_val.shape,y_val.shape
bayes = GaussianNB()
bayes.fit(x_train,y_train)
print(bayes.score(x_train,y_train))
y_predict = bayes.predict(x_val)
print(accuracy_score(y_val,y_predict))