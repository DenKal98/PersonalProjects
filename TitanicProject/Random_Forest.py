#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:59:37 2019

@author: denkal
"""

import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
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
rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=3,max_features=0.5,n_jobs=-1)
rf.fit(x_train,y_train)
print(rf.score(x_train,y_train))

y_predict = rf.predict(x_val)
print(accuracy_score(y_val,y_predict))
