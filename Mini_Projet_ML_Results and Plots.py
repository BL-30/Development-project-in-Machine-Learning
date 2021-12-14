#!/usr/bin/env python
# coding: utf-8

# In[2]:


from Mini_Projet_Fonctions.py import ML_method


# In[ ]:


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


# In[ ]:


plt.figure(figsize=(15,8))
plt.bar(list(accuracy.keys()), accuracy.values())
plt.title('Accuracy of different ML methods')
plt.ylabel('Accuracy score')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
plt.bar(list(precision.keys()), precision.values())
plt.title('Precision of different ML methods')
plt.ylabel('Precision score')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
plt.bar(list(recall.keys()), recall.values())
plt.title('Recall of different ML methods')
plt.ylabel('Recall score')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
plt.bar(list(fscore.keys()), fscore.values())
plt.title('FScore of different ML methods')
plt.ylabel('FScore score')
plt.show()

