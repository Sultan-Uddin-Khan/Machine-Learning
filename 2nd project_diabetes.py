#!/usr/bin/env python
# coding: utf-8

# In[50]:


# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import sin
import pandas as pd
from pylab import rcParams
import numpy as np


# In[10]:


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(url, names=names)


# In[11]:


# shape
print(data.shape)


# In[12]:


# head
print(data.head(20))


# In[13]:


# descriptions
print(data.describe())


# In[15]:


# class distribution
print(data.groupby('class').size())


# In[18]:


# box and whisker plots
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.rcParams['figure.figsize'] = [30,5]
pyplot.show()


# In[21]:


# histograms
data.hist()
pyplot.rcParams['figure.figsize'] = [50,10]
pyplot.show()


# In[24]:


# scatter plot matrix
scatter_matrix(data)
pyplot.rcParams['figure.figsize'] = [17,5]
pyplot.show()


# In[25]:


# plotting the density plot 
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.rcParams['figure.figsize'] = [40,5]
pyplot.show()


# In[27]:


#Correlation matrix plot
correlations = data.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# In[32]:


# create line plot
data.sort_values(by=["pres"], inplace = True)
rcParams['figure.figsize'] = 12, 5
sns.lineplot(x=data['pres'].astype(str),y=data['class'],color='r')
plt.xlabel('BloodPressure')
plt.ylabel('class')
plt.tick_params(axis='x',labelsize=8)


# In[33]:


sns.scatterplot(data=dataset, x="BMI", y="Insulin")


# In[35]:


name = data['plas']
 
# Figure Size
fig = plt.figure(figsize =(10, 7))
 
# Horizontal Bar Plot
glucose1=plt.bar(name[0:100], height=0.4, width=1,label='0-100')
glucose2=plt.bar(name[101:150], height=0.4, width=1,label='101-150')
glucose3=plt.bar(name[151:200], height=0.4, width=1,label='151-200')
# Show Plot
plt.legend(handles=[glucose1, glucose2,glucose3])
plt.xlabel('Glucose')
plt.show()


# In[47]:


#Split-out validation dataset
array = data.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


# In[51]:


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

