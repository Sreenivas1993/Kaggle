# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 18:22:27 2017

@author: Sreenivas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import tree
from sklearn.preprocessing import Imputer

if __name__=="__main__":
    #Read dataset
    test=pd.read_csv('test.csv')
    train=pd.read_csv('train.csv')
    gender=pd.read_csv('gender_submission.csv')
    #Checking for missing values
    missingdat=train.isnull().sum()
    """print(missingdat)"""
    # There are missing values in Age and Cabin. So to replace missing values use Imputer library
    imputerage=Imputer(missing_values='NaN',axis=0)
    imputerage=imputerage.fit(train[['Age']])
    train['Age']=imputerage.transform(train[['Age']]).ravel()
    
    #USE value_counts to get the most frequent item in column to replace null value
    train['Cabin']=train['Cabin'].fillna(train['Cabin'].value_counts().index[0])
    
    #Encoding Categorical data
    from sklearn.preprocessing import LabelEncoder
    labelencodersex=LabelEncoder()
    train['Sex']=labelencodersex.fit_transform(train['Sex'])
    
    #fitting tree classifier
    target=train["Survived"].values
    features=train[["Pclass", "Sex", "Age", "Fare"]].values
    my_tree_one=tree.DecisionTreeClassifier()
    my_tree_one=my_tree_one.fit(features,target)
   
    #Calculating importance of features in model created
    """print(my_tree_one.feature_importances_)
    print(my_tree_one.score(features,target))"""
    
    #prediction
    #checking for missing values in test set
    missingtest=test.isnull().sum()
    """print(missingtest)"""
    test['Fare']=test['Fare'].fillna(test['Fare'].median())
    test['Sex']=labelencodersex.fit_transform(test['Sex'])
    imputerage=imputerage.fit(test[['Age']])
    test['Age']=imputerage.transform(test[['Age']]).ravel()
    test_features=test[["Pclass", "Sex", "Age", "Fare"]].values
    prediction=my_tree_one.predict(test_features)
    PassengerId=np.array(test['PassengerId']).astype(int)
    solution=pd.DataFrame(prediction,PassengerId,columns=["Survived"])
    solution.to_csv("solution.csv",index_label=["PassengerId"])
    
    
   