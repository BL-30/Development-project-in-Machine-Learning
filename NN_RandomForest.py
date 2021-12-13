from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def neural_network(data,n):
    X = data.iloc[:,0:data.shape[1]-1]
    y = data.iloc[:,data.shape[1]-1]
    
    new_clean(X)
    if type_data == 'kidney' :
        X = pd.get_dummies(X) # only for kidney
    preprocessing.normalize(X, axis = 0)
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    sel.fit_transform(X)

    acc = np.zeros(n)

    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
        accu=clf.score(X_test, y_test)
        acc[i] = accu
        
    return np.mean(acc,axis=0)
  
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

    return df
  
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
  
  def random_forest(data,n):
    X = data.iloc[:,0:data.shape[1]-1]
    y = data.iloc[:,data.shape[1]-1]
    
    

    new_clean(X)
    if type_data == 'kidney' :
        X = pd.get_dummies(X) # only for kidney
    preprocessing.normalize(X, axis = 0)
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    sel.fit_transform(X)

    

    op_max_depth= optimal_max_depth(X)

    acc = np.zeros(n)


    for i in range(n):
        rfc = RandomForestClassifier(n_estimators=100,max_depth=op_max_depth)
        rfc.fit(X_train, y_train)
        accu=rfc.score(X_test, y_test)
        acc[i] = accu
        
    return np.mean(acc,axis=0)
