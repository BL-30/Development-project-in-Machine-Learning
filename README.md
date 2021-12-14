Status

<h1 align="center">Development project in Machine Learning</h1>
<h3 >Nom des étudiants : Cyprien Le Rouge de Guerdavid, Gabriel Gros, Laurine Marty, Baptiste Loyer</h2>
</h3>

<h4 align="center">
	🚧 Development project in Machine Learning 🚀 Under construction...  🚧
</h4>

<hr>

<p align="center">
  <a href="#dart-about">About</a> &#xa0; | &#xa0; 
  <a href="#rocket-Objectives">Objectives</a> &#xa0; | &#xa0;
  <a href="#white_check_mark-requirements">Requirements</a> &#xa0; | &#xa0;
  <a href="#checkered_flag-starting">Starting</a> &#xa0; | &#xa0;
</p>

<br>

## :rocket: Objectives

The following skills/tasks were aimed by this project:

-  Develop good programming practices
-  Use standard development tools
-  Get used to collaborative work
-  Work on Machine-Leaning datasets

- Write the Python functions implementing the workflow in one single .py file.
- Apply the workflow onto the two datasets, using either a Python script or a notebook.
- Important: Your .py file containing the functions must be the same when applied to one or the other dataset

## :dart: About
I Present the project:
Our Project in machine learning aim to apply a Machine Learning model onto two different datasets :
- Binary Classification
Banknote Authentication Dataset: https://archive.ics.uci.edu/ml/datasets/banknote+authentication
- Chronic Kidney Disease: https://www.kaggle.com/mansoordaku/ckdisease


## DATA 📊
Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.
1. variance of Wavelet Transformed image (continuous)
2. skewness of Wavelet Transformed image (continuous)
3. curtosis of Wavelet Transformed image (continuous)
4. entropy of image (continuous)
5. class (integer)

Chronic KIdney Disease dataset
Data has 25 features which may predict a patient with chronic kidney disease

The data was taken over a 2-month period in India with 25 features ( eg, red blood cell count, white blood cell count, etc). The target is the 'classification', which is either 'ckd' or 'notckd' - ckd=chronic kidney disease. There are 400 rows

The data needs cleaning: in that it has NaNs and the numeric features need to be forced to floats. Basically, we were instructed to get rid of ALL ROWS with Nans, with no threshold - meaning, any row that has even one NaN, gets deleted.

I To complete

The methods we used:

Methods
K-means
Hard clustering classification method which aims to partition the data into clusters. It affects each sample to the class which has the closest centroid.

Advantages : easy to implement
Drawbacks : can’t deal with non convex problems

Bayesian GMM
It is a statistical classifier involving Bayes Theorem on conditional probabilities. 

Therefore, for a given sample of data, we can calculate the probability of the sample to be part of a class Ck.


Kernel methods and SVM
Kernels are a class of algorithms which maps data into high dimensional feature space to classify it more easily. The kernel trick is then to find solutions to compute efficiently the higher-order model.

Advantages : keep the advantages of simple models while attaining the advantages of more complex models to reduce bias and variance.
Drawbacks : needs to find the good kernel and its associated parameters.

Neural networks
Composition of several functions in neurons and layers which quickly leads to very complex functions. Therefore, input as output vectors are fully connected with all the layers of the network. 

Advantages : presents the best performances and derivatives can be computed easily.

Random forests
Ensemble learning method which constructs several decision trees for training.The output is then given by the class selected by most trees.

Advantages : easily understandable.

I BIS the main development steps:
Visualization of the data

In classification problems the visualization could be a good first introduction to the dataset. It can be used to see whether certain clusters are particularly detached or not. Moreover it is a way for the data-scientist to use his intuition over certain parameters of the data processing or to develop an heuristic approach to the problem.  

However, most of the time the number of features of the datasets unable to visualize it directly. Therefore, it is necessary to go through a dimension reduction phase before making a display.
One of the most common techniques for dimension reduction is the Principal Component Analysis (PCA).
PCA is affected by scale so we need to scale the features in our data before applying PCA.
By analyzing the second data set : “kidney disease”, where data has 25 features and which may predict a patient with chronic disease, we notice that there could have two category of features : the quantitative (basically a numeric value) or a categorical data (which can be represented by a string, a number or a Boolean). This duality in the data type is an obstacle to the application of PCA. Indeed, the mathematical formalism of the scikit-learn method we previously intend to use, is adapted only for the first type of values.
Similarly, there exists another method  called “Multiple Correspondence Analysis” which has exactly the same goal but which is only applicable to categorical data.
But, in order to adapt our process to all types of data set (with only quantitative data, only categorical data or even a mix of the two types) we have to find a new type of algorithm.
“Prince” is a Python library which has a module FAMD for “Factor Analysis for Mixed Data'' which aims to apply the principle of PCA and MCA to a data set composed of the two types of data.
 
This library is under the MIT license but can be used as our type of project as an open-source algorithm.

![image](https://user-images.githubusercontent.com/91438136/146048612-bf4a69d8-90ee-4b32-b514-31a41f5a7c51.png) ![image](https://user-images.githubusercontent.com/91438136/146048723-bf019332-eb9d-4492-b1bb-0c9302da828a.png)

Preparation of the data 

First of all, in order to apply methods properly we have to clean the dataset. Indeed, some datasets are partially filled and some value remains associated with ‘Nan’. 
We separated the cleaning process in two parts : first we treat the numeric data by replacing the missing values by the median of the column. Then for the non numeric data we replace the missing values by the most frequently occuring value. 

Second, we have to transform data so that models will be able to treat them. For every model numeric values are the most common way to represent data. Thus we use a label encoder to associate each category to a number in the categorical data columns. 

Then, in order to avoid bias we decided to normalize our data set. We transform data to obtain the unit norm for each column. 

Finally, for the Neural Network only, we apply a features selection function. Neural Networks are the models that require the biggest ressources to be implemented. By reducing the number of features we also reduce the working time. Moreover, neural networks are not easily interpretable, so feature selection does not impact on this aspect in contrast to what it could have done on random forests, which have the main advantage of being easily readable with the input features. 
We apply the Variance Threshold algorithm : it removes all low-variance features.

Training models and cross validation 

In order to facilitate the comparison of the model we implement each training process, followed by a testing phase in separated functions which take in argument the name of the dataset and the number of times we want to repeat the process to have an optimal statistical view on the method performance. 
Each time we separate the dataset treated randomly in two parts : a training set and a testing set, we train the model with the training set. Then we apply the model to the testing set and we return the information about performances : the accuracy of the model and for each class : the precision and the recall ratios. 
By repeating the training/testing process a large number of times on randomly chosen parts of the dataset we are able to make the mean of the obtained results to have an information representative of the performance.  
  
Compare results 

By returning the same ratios for each method we can easily compare the performances of them on each dataset. Performances can, indeed, be different according to the type of data : in the case where there is only numeric data (Bank dataset) or if the dataset is a mix of both quantitative and categorical data (Kidney disease dataset). 
Histograms appear as a lisible way to represent scores and compare them.  

II Show and comment your results:
## Here are our results concerning the first dataset named :  data_banknote 📈
On this graph we can observe that the greater the depth is, the better is our accuracy concerning the first dataset
![image](https://user-images.githubusercontent.com/91438136/146052237-4fe619e6-e154-483e-86f0-1ef3b417d5e9.png)

On this barchart we can observe that some methods like clustering or kernel sigmoid are not quite appropriate to run a good accuracy score concerning the first dataset
![image](https://user-images.githubusercontent.com/91438136/146089919-3bb77a4c-2805-4df8-bb58-144b10732e8c.png)

On this barchart we can observe that some methods like clustering or kernel sigmoid are not quite appropriate to run a good precision score concerning the first dataset
![image](https://user-images.githubusercontent.com/91438136/146089808-c5bf1d5b-ca79-4e72-8308-96ce0e1a14ac.png)

On this barchart we can observe that some methods like clustering or kernel sigmoid are not quite appropriate to run a good recall score concerning the first dataset
![image](https://user-images.githubusercontent.com/91438136/146090001-f3360e33-8952-4ed4-86fa-1164f896ee61.png)

On this barchart we can observe that some methods like clustering or kernel sigmoid are not quite appropriate to run a good fscore score concerning the first dataset
![image](https://user-images.githubusercontent.com/91438136/146090086-300c2b9d-e753-4c08-b8ac-bfaafc151b03.png)


Finally 5 methods makes very-well estimation for this dataset : kernel_linear, kernel_polynomial, kernel_gaussian, neural_networks and random_forests.



## Here are our results concerning the second dataset named :  kidney disease 📈


![image](https://user-images.githubusercontent.com/91438136/146053531-075b0fe7-a41e-4225-ba32-32b131b10846.png)

![image](https://user-images.githubusercontent.com/91438136/146053594-8042b826-861e-45f9-86df-0746e9ed0557.png)

![image](https://user-images.githubusercontent.com/91438136/146090473-f9158a15-c167-4d54-baf5-3fe97821bb6b.png)

![image](https://user-images.githubusercontent.com/91438136/146090529-c3c662ca-adcd-49d6-87f7-67ecdc4307fe.png)
![image](https://user-images.githubusercontent.com/91438136/146090602-b5ef15f7-93fa-4f8a-aa88-5f9ee80b2cb3.png)


On those barcharts, as well as the precedent dataset, sigmoid and clustering are not appropriated.
We notice also that neural networks is less perfomant for this dataset
Finally 4 methods makes very-well estimation for this dataset : kernel_linear, kernel_polynomial, kernel_gaussian, random_forests.

## What we think are good programming practices

We think that the best way to have an effective code is to split our code into multiple functions.

- Simplification of Code:
We were assigned a complex problem, the first idea we had resulted in a complicated solution with an overall lengthy and clumsy code block. Although that is not a totally negative aspect because we figured out one of the approaches to solve the problem, we realize that it is not enough.We dwelled deep and explore more unique methods to obtain the solutions.

- Planification of our Approach:
Before we dive straight into the problem, we planned our approach accordingly. It is not always the best idea for us to jump into coding for a complex problem. We thought it will be a better proposition to effectively plan our course of action and dwelled on the specifics.

- Indentation
Formatting and indentation are necessary to organize our code. Ideal coding formatting and indentation include correct spacing, line length, wraps and breaks. By employing indentations, white-spacing, and tabs within the code, we ensure our code is readable and organized.
IV In appendix, provide the logs of your git repository and your code.

In this project we aim to collaboratively implement this workflow and apply it to different ML problems/datasets

Firstly, we import the dataset
2. Clean the data, perform pre-processing
Replace missing values by average or median values
Center and normalize the data
3. Split the dataset
Split between training set and test set
Split the training set for cross-validation
4. Train the model (including feature selection)
5. Validate the model

## Plan / Rules
We first clean the dataset (handle missing values and
categorical values)
Then, we implement feature selection: bruteforce, by looking
at correlations, from an ACP (for classification), by using
Ridge regression (for linear regression), etc.
We do not forget to save a part of our dataset as our test set.
It will not be used for training, but only to assess the quality
of our method.
We also used cross-validation to adjust the method
(choice of the kernel, feature selection, etc.)
We try to automate your process as much as possible

- We first commit our code in multiple .py, and then merged them in order to write the Python functions implementing the workflow in one
single .py file.
- We applied the workflow onto the two datasets, using either a Python script or a notebook.
- Important: Our .py file containing the functions is the same when applied to one or the other dataset
- Each student of the group wrote at least one function.
- We indicate the writer of each function in comment
- We also created a git repository for our group: https://redmine-df.telecom-bretagne.eu/
- Log of our Github Repository : https://github.com/BL-30/Development-project-in-Machine-Learning


## :white_check_mark: Requirements

Before starting :checkered_flag:, you need to have [Git](https://git-scm.com), and [VsCode](https://code.visualstudio.com/) installed.

## :checkered_flag: Starting

```bash
# Clone this project
$ git clone https://github.com/https://github.com/BL-30/Development-project-in-Machine-Learning

```
# Appendix : 

# Functions code : 

## Imports

	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	import random as rnd

	from sklearn import preprocessing
	from sklearn.preprocessing import OneHotEncoder, LabelEncoder

	from sklearn.model_selection import train_test_split

	from sklearn.cluster import KMeans

	from sklearn.metrics import accuracy_score
	from sklearn.metrics import precision_recall_fscore_support as score

	from sklearn.feature_selection import VarianceThreshold
	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import chi2
	from sklearn.feature_selection import SelectFromModel

	from sklearn.svm import LinearSVC
	from sklearn.svm import SVC

	from sklearn.neural_network import MLPClassifier
	from sklearn.model_selection import train_test_split

	from sklearn.ensemble import RandomForestClassifier

	from sklearn.datasets import load_iris
	from sklearn.naive_bayes import GaussianNB

	from sklearn.kernel_approximation import RBFSampler
	from sklearn.linear_model import SGDClassifier

## Choosing Data : (Made by Gabriel)

	def choose_data(b):
	    if b:
		data = pd.read_csv('data_banknote_authentication.txt')
	    else:
		data =  pd.read_csv('kidney_disease.csv')
	    return data

## Clean Data : (Made by Cyprien)

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

	    return df

## Optimal depth for Random Forest : (Made by Laurine)
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

## Machine Learning Method Application : (Made by Laurine, Baptiste, Gabriel, and Cyprien)
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
		op_max_depth = optimal_max_depth(data)

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
    

# Results and Plots :

## Calculations
	from Mini_Projet_Fonctions.py import ML_method

	methods = ['clustering', 'bayesian', 'kernel_linear', 'kernel_poly', 'kernel_gaussian', 'kernel_sigmoid','neural_network', 'random_forest']

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

## Plot Accuracy
	plt.figure(figsize=(15,8))
	plt.bar(list(accuracy.keys()), accuracy.values())
	plt.title('Accuracy of different ML methods')
	plt.ylabel('Accuracy score')
	plt.show()

## Plot Precision
	plt.figure(figsize=(15,8))
	plt.bar(list(precision.keys()), precision.values())
	plt.title('Precision of different ML methods')
	plt.ylabel('Precision score')
	plt.show()

## Plot Recall
	plt.figure(figsize=(15,8))
	plt.bar(list(recall.keys()), recall.values())
	plt.title('Recall of different ML methods')
	plt.ylabel('Recall score')
	plt.show()

## Plot FScore
	plt.figure(figsize=(15,8))
	plt.bar(list(fscore.keys()), fscore.values())
	plt.title('FScore of different ML methods')
	plt.ylabel('FScore score')
	plt.show()




