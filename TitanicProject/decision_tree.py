#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:53:35 2019

@author: denkal
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score




train_df = pd.read_csv("data/num_train.csv")


#Making the training dataset for the model
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]

def split_values(x,y): return x[:y], x[y:]
count = 60
number = len(X_train) - count 
x_train, x_val = split_values(X_train,number)
y_train, y_val = split_values(Y_train,number)
x_train.shape,y_train.shape,x_val.shape,y_val.shape
dt = DecisionTreeClassifier(min_samples_leaf = 3, max_features = 0.5)
dt.fit(x_train,y_train)
print(dt.score(x_train,y_train))

y_predict = dt.predict(x_val)
print(accuracy_score(y_val,y_predict))