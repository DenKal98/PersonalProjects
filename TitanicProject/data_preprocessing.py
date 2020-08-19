#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:50:05 2019

@author: denkal
"""

import pandas as pd 
import numpy as np 

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

#This creates a df to show how many null values there are in the columns
total = train_df.isnull().sum().sort_values(ascending=False)
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

#Dropping passenger ID and Cabin from the dataset
train_df = train_df.drop(['PassengerId'], axis = 1)
train_df = train_df.drop(['Cabin'],axis = 1)
train_df = train_df.drop(["Ticket"], axis = 1)
train_df = train_df.drop(["Name"], axis = 1)
test_df = test_df.drop(['PassengerId'], axis = 1)
test_df = test_df.drop(['Cabin'],axis = 1)
test_df = test_df.drop(["Ticket"], axis = 1)
test_df = test_df.drop(["Name"], axis = 1)



#Dealing with the Null values 
df = [train_df, test_df]

#Finding the mean and standard deviation fo age so i can fill in the null values with the most relevant values 
for data in df:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = data["Age"].isnull().sum()
    #Computing random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    #Fill NaN values in Age column with random values generated
    age_slice = data["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    data["Age"] = age_slice
    data["Age"] = train_df["Age"].astype(int)
train_df["Age"].isnull().sum()

#Filling in NaN values in embarked with most common port as it is only 2 values
desc = train_df["Embarked"].describe()

common_value = 'S'

for data in df:
    data['Embarked'] = data['Embarked'].fillna(common_value)
    
#Changing Male and Female to 1 and 0

values = {"male" : 1, "female" : 0}

for data in df:
    data['Sex'] = data['Sex'].map(values)

#Changing the Embarked C S Q to 0 1 2

values = {"C" : 0, "S" : 1, "Q" : 2}

for data in df:
    data["Embarked"] = data["Embarked"].map(values)

#test_df = test_df.drop(test_df.index[152])

#test_df.to_csv("num_test.csv")

