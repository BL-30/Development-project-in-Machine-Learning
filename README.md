Status

<h1 align="center">Development project in Machine Learning</h1>

Status

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

the methods we used:
We used different methods, including some functions already implemented in the Sk Learn library. We code 

the main development steps:
We implement the code collaborately on git hub, and we were used to meet very often to work on this project. We used different branches based on our names_dev. 
After that, we merged the different branches together.

II Show and comment your results:

------------------ RESULTS HERE ------------------

III Include one part describing what you think are good programming practices

We think that the best way to have an effective code is to split our code into multiple functions.
- ADHERE TO YOUR STYLE GUIDE
- Simplification of Code:
We were assigned a complex problem, the first idea we had resulted in a complicated solution with an overall lengthy and clumsy code block. Although that is not a totally negative aspect because we figured out one of the approaches to solve the problem, we realize that it is not enough.We dwelled deep and explore more unique methods to obtain the solutions.

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

You should first clean the dataset (handle missing values and
categorical values)
You may implement feature selection: bruteforce, by looking
at correlations, from an ACP (for classification), by using
Ridge regression (for linear regression), etc.
Do not forget to save a part of your dataset as your test set.
It will not be used for training, but only to assess the quality
of your method.
You may also use cross-validation to adjust the method
(choice of the kernel, feature selection, etc.)
You should automate your process as much as possible

Create a git repository for your group: https://redmine-df.telecom-bretagne.eu/
- Write the Python functions implementing the workflow in one
single .py file.
- Apply the workflow onto the two datasets, using either a Python script or a notebook.
- Important: Your .py file containing the functions must be the same when applied to one or the other dataset
- Each student of the group should write at least one function.
Indicate the writer of each function in comment


## :white_check_mark: Requirements

Before starting :checkered_flag:, you need to have [Git](https://git-scm.com), and [VsCode](https://code.visualstudio.com/) installed.

## :checkered_flag: Starting

```bash
# Clone this project
$ git clone https://github.com/https://github.com/BL-30/Development-project-in-Machine-Learning

```
