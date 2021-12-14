#!/usr/bin/env python
# coding: utf-8

# In[1]:


#fichier fonction

#poster un fichier plot

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

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# In[2]:


def choose_data(b):
    if b:
        data = pd.read_csv('data_banknote_authentication.txt')
    else:
        data =  pd.read_csv('kidney_disease.csv')
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


# In[4]:


def optimal_max_depth(data):
    X = data.iloc[:,0:data.shape[1]-1]
    y = data.iloc[:,data.shape[1]-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    score=[]
    md=[]
    for i in range(1,21):
        rfc = RandomForestClassifier(n_estimators=100,max_depth=i)
        rfc.fit(X_train, y_train)
        score.append(rfc.score(X_test,y_test))
        md.append(i)
    print("Maximum performance is archieved with a maximal depth of :  ", score.index (max (score))+1)
    plt.plot(md,score) 
    plt.title("Evolution of accuracy according to the maximal depth of the Random Forest generated")
    return score.index (max (score))+1


# In[5]:


def ML_method(method, data, n, type_data):
    X = data.iloc[:,0:data.shape[1]-1]
    y = data.iloc[:,data.shape[1]-1]

    new_clean(X)
    if type_data == 'kidney' :
        le = LabelEncoder()
        X = X.apply(le.fit_transform)
    preprocessing.normalize(X, axis = 0)

    acc, precision, recall, fscore, support = np.zeros(n), [], [], [], []
    if method == 'random_forest':
         op_max_depth= optimal_max_depth(data)
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        if method == 'clustering':
            kmeans = KMeans(n_clusters=2,n_init=1,init='k-means++').fit(X_train,y_train)
            y_pred = kmeans.predict(X_test)
        elif method == 'bayesian':
            gnb = GaussianNB()
            y_pred = gnb.fit(X_train, y_train).predict(X_test)
        elif method == 'kernel_linear':
            svclassifier = SVC(kernel='linear')
            svclassifier.fit(X_train, y_train)
            y_pred = svclassifier.predict(X_test)
        elif method == 'kernel_poly':
            svclassifier = SVC(kernel='poly')
            svclassifier.fit(X_train, y_train)
            y_pred = svclassifier.predict(X_test)
        elif method == 'kernel_gaussian':
            svclassifier = SVC(kernel='rbf')
            svclassifier.fit(X_train, y_train)
            y_pred = svclassifier.predict(X_test)
        elif method == 'kernel_sigmoid':
            svclassifier = SVC(kernel='sigmoid')
            svclassifier.fit(X_train, y_train)
            y_pred = svclassifier.predict(X_test)
        elif method == 'neural_network':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
        elif method == 'random_forest':
            
            rfc = RandomForestClassifier(n_estimators=100,max_depth=op_max_depth)
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)
            
        acc[i] = accuracy_score(y_test,y_pred)
        prec_temp, rec_temp, fs_temp, sup_temp = score(y_test, y_pred)
        precision.append(prec_temp)
        recall.append(rec_temp)
        fscore.append(fs_temp)
        support.append(sup_temp)
        
    return np.mean(acc,axis=0), np.mean(precision), np.mean(recall),np.mean(fscore), np.mean(support, axis=0)    


# In[30]:


methods = ['clustering', 'bayesian', 'kernel_linear', 'kernel_poly', 'kernel_gaussian', 'kernel_sigmoid', 'neural_network', 'random_forest']

type_data = 'bank'
data_bank = choose_data(True)
n=30

accuracy , precision, recall, fscore, support = {},{},{},{},{}

for method in methods:
    acc, prec, rec, fs, supp = ML_method(method, data_bank, n, type_data)
    accuracy[method] = acc
    precision[method] = prec
    recall[method] = rec
    fscore[method] = fs
    support[method] = supp
    


# In[31]:



plt.figure(figsize=(15,8))
plt.bar(list(accuracy.keys()), accuracy.values())
plt.title('Accuracy of different ML methods')
plt.ylabel('Accuracy score')
plt.show()


# In[32]:



plt.figure(figsize=(15,8))
plt.bar(list(precision.keys()),precision.values())
plt.title('Precision of different ML methods')
plt.ylabel('Accuracy score')
plt.show()


# In[33]:


plt.figure(figsize=(15,8))
plt.bar(list(recall.keys()), recall.values())
plt.title('Recall of different ML methods')
plt.ylabel('Accuracy score')
plt.show()


# In[34]:



plt.figure(figsize=(15,8))
plt.bar(list(fscore.keys()), fscore.values())
plt.title('Fscore of different ML methods')
plt.ylabel('Accuracy score')
plt.show()


# In[36]:



plt.figure(figsize=(15,8))
plt.bar(list(support.keys()), accuracy.values())
plt.title('Support of different ML methods')
plt.ylabel('Accuracy score')
plt.show()

