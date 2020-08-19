# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:39:23 2020

@author: DKaylan
"""

import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

df = pd.concat([train,test])