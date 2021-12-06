#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# In[2]:


def choose_data(b):
    if b:
        data = pd.read_csv(r'C:\Users\gabgr\OneDrive\Bureau\Machine Learning\Projet ML\data_banknote_authentication.txt')
    else:
        data =  pd.read_csv(r'C:\Users\gabgr\OneDrive\Bureau\Machine Learning\Projet ML\kidney_disease.csv')
    return data


# In[3]:


def new_clean(df):
    # select numerical columns
    df_numeric = df.select_dtypes(include=[np.number])
    numeric_cols = df_numeric.columns.values

    # select non-numeric columns
    df_non_numeric = df.select_dtypes(exclude=[np.number])
    non_numeric_cols = df_non_numeric.columns.values


    #imput missing numerical values
    for col in numeric_cols:
        missing = df[col].isnull()
        num_missing = np.sum(missing)
        if num_missing > 0:  # impute values only for columns that have missing values
            med = df[col].median() #impute with the median
            df[col] = df[col].fillna(med)
        
    #imput missing non numerical values        
    for col in non_numeric_cols:
        missing = df[col].isnull()
        num_missing = np.sum(missing)
        if num_missing > 0:  # impute values only for columns that have missing values
            mod = df[col].describe()['top'] # impute with the most frequently occuring value
            df[col] = df[col].fillna(mod)
        
    #verification of cleaning : if the result is 0 -> succeed        
    #print('verification of cleaning : if the result is 0 -> succeed : ',df.isnull().sum().sum())

    # dropping duplicates by considering all columns other than ID
    #cols_other_than_id = list(df.columns)[1:]
    #df.drop_duplicates(subset=cols_other_than_id, inplace=True)
    return df


# In[68]:


def bayesian(data,n,type_data):
    X = data.iloc[:,0:data.shape[1]-1]
    y = data.iloc[:,data.shape[1]-1]
    #print("Before transform ",X.shape)

    new_clean(X)
    #print("After cleaning ",X)
    
    if type_data == 'kidney' :
        le = LabelEncoder()
        X = X.apply(le.fit_transform)
        #hotencoder = OneHotEncoder()
        #X = hotencoder.fit_transform(X)
        #labelencoder = LabelEncoder()
        #y = labelencoder.fit_transform(y)
        #X = pd.get_dummies(X) # only for kidney
    #print("After encoding ",X.shape)
    preprocessing.normalize(X, axis = 0)

    #sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    #sel.fit_transform(X)
    #X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(X)

    acc = np.zeros(n)

    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3)
        
        gnb = GaussianNB()
        y_pred = gnb.fit(X_train, y_train).predict(X_test)
        
        #kmeans = KMeans(n_clusters=2,n_init=1,init='k-means++').fit(X_train,y_train)
        #centers = kmeans.cluster_centers_

        #y_pred = kmeans.predict(X_test)
        acc[i] = accuracy_score(y_test,y_pred)

    return np.mean(acc,axis=0)


# In[69]:


def kernel(data,n,type_kernel,type_data):
    X = data.iloc[:,0:data.shape[1]-1]
    y = data.iloc[:,data.shape[1]-1]

    new_clean(X)
    if type_data == 'kidney' :
        le = LabelEncoder()
        X = X.apply(le.fit_transform)
        #X = pd.get_dummies(X) # only for kidney
    preprocessing.normalize(X, axis = 0)
    #sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    #sel = LabelEncoder()
    #sel.fit_transform(X)
    #X_new = SelectKBest(chi2, k=2).fit_transform(X, y)

    acc = np.zeros(n)

    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        #svclassifier = SVC(kernel='poly', degree=8)
        #svclassifier = SVC(kernel='rbf')
        #svclassifier = SVC(kernel='sigmoid')
        svclassifier = SVC(kernel=type_kernel)
        svclassifier.fit(X_train, y_train)
        y_pred = svclassifier.predict(X_test)
        acc[i] = accuracy_score(y_test,y_pred)
        
    return np.mean(acc,axis=0)


# In[70]:


def clustering(data,n,type_data):
    X = data.iloc[:,0:data.shape[1]-1]
    y = data.iloc[:,data.shape[1]-1]
    #print(y)
    #print("X v1",X)
    #print("Y",y)
    X = new_clean(X)
    
    if type_data == 'kidney' :
        le = LabelEncoder()
        X = X.apply(le.fit_transform)
        #y = y.apply(le.fit_transform)
        #X = pd.get_dummies(X) # only for kidney
        #y = pd.get_dummies(y)
   
    
    #print("X v2",X)
    #print("clean ok","\n")
    
    preprocessing.normalize(X, axis = 0)
    #print("norm ok","\n")
    
    #sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    #sel.fit_transform(X)
    #X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
    #print("feat ok","\n")
    
    acc = np.zeros(n)

    for i in range(n):
        #print("for ok","\n")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        #print("split ok","\n")

        kmeans = KMeans(n_clusters=2,n_init=1,init='k-means++').fit(X_train,y_train)
        #centers = kmeans.cluster_centers_
        #print("train ok","\n")

        y_pred = kmeans.predict(X_test)
        #print(len(y_pred))
        
        if type_data == 'kidney' :
            A = [str(pred) for pred in y_pred]
            for j in range(len(A)):
                if A[j]=='1':
                    A[j] = 'ckd'
                else:
                    A[j] = 'notckd'
                    
        else:
            A = y_pred
        
        #print("test ok",A)
        acc[i] = accuracy_score(y_test,A)

    return np.mean(acc,axis=0)


# In[71]:


type_data = 'bank'
data_bank = choose_data(True)

n=30

print("Data Banknote : ","\n")
print("clustering accuracy = ", clustering(data_bank,n,type_data))
print("bayesian accuracy = ", bayesian(data_bank,n,type_data))
print("kernel linear accuracy = ", kernel(data_bank,n,'linear',type_data))
print("kernel polynomial accuracy = ", kernel(data_bank,n,'poly',type_data))
print("kernel sigmoid accuracy = ", kernel(data_bank,n,'sigmoid',type_data))
print("kernel gaussian accuracy = ", kernel(data_bank,n,'rbf',type_data))


# In[73]:


type_data = 'kidney'
data_kid = choose_data(False)

n=30

print("Data Kidney : ","\n")
print("clustering accuracy = ", clustering(data_kid,n,type_data))
print("bayesian accuracy = ", bayesian(data_kid,n,type_data))
print("kernel linear accuracy = ", kernel(data_kid,n,'linear',type_data))
print("kernel polynomial accuracy = ", kernel(data_kid,n,'poly',type_data))
print("kernel sigmoid accuracy = ", kernel(data_kid,n,'sigmoid',type_data))
print("kernel gaussian accuracy = ", kernel(data_kid,n,'rbf',type_data))


# In[ ]:




