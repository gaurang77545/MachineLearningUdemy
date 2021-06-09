# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
print("hello world")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv("Data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:, 0] = labelencoder_X.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test =  train_test_split(x, y, test_size= 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

