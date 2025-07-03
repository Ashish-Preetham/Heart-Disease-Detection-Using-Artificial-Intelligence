#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('C:\\Users\\VADITHYA SUCHITHRA\\OneDrive\\Desktop\\mini project\\heart.csv')
print(data)


# In[2]:


data.head(5)


# In[3]:


print("(Rows, columns):  " + str(data.shape))
data.columns
data.nunique(axis=0)


# In[4]:


data.describe().T


# In[5]:


corr = data.corr()
plt.subplots(figsize=(12,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))


# In[6]:


subData = data[['age','trestbps','chol','thalach','oldpeak']]

sns.pairplot(subData)


# In[7]:


plt.figure(figsize=(12,8))

sns.violinplot(x= 'target', y= 'oldpeak',hue="sex", inner='quartile',data= data )

plt.title("Thalach Level vs. Heart Disease",fontsize=20)

plt.xlabel("Heart Disease Target", fontsize=16)

plt.ylabel("Thalach Level", fontsize=16)


# In[8]:


plt.figure(figsize=(12,8))

sns.boxplot(x= 'target', y= 'thalach',hue="sex", data=data )

plt.title("ST depression Level vs. Heart Disease", fontsize=20)

plt.xlabel("Heart Disease Target",fontsize=16)

plt.ylabel("ST depression induced by exercise relative to rest", fontsize=16)


# In[9]:


pos_data = data[data['target']==1]
pos_data.describe()
pos_data = data[data['target']==0]
pos_data.describe().T


# In[10]:


from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
model1 = LogisticRegression(random_state=1) 
model1.fit(x_train, y_train)
y_pred1 = model1.predict(x_test)
print(classification_report(y_test, y_pred1))


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
X = data.iloc[:, :-1].values
y= data.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
model2 = KNeighborsClassifier() # get instance of model
model2.fit(x_train, y_train) # Train/Fit model
y_pred2 = model2.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred2))


# In[12]:


from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

model3 = SVC(random_state=1) # get instance of model

model3.fit(x_train, y_train) # Train/Fit model

y_pred3 = model3.predict(x_test) # get y predictions

print(classification_report(y_test, y_pred3)) 


# In[13]:


from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

model4 = GaussianNB() # get instance of model

model4.fit(x_train, y_train) # Train/Fit model

y_pred4 = model4.predict(x_test) # get y predictions

print(classification_report(y_test, y_pred4))


# In[14]:


from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

model5 = RandomForestClassifier(random_state=1)# get instance of model

model5.fit(x_train, y_train) # Train/Fit model

y_pred5 = model5.predict(x_test) # get y predictions

print(classification_report(y_test, y_pred5))


# In[15]:


# get importance

importance = model5.feature_importances_

# summarize feature importance

for i,v in enumerate(importance):

   print('Feature: %0d, Score: %.5f' % (i,v))


# In[36]:


from sklearn.metrics import accuracy_score
print("accuracy of Logistic regression model:",accuracy_score(y_test,y_pred1))
print("accuracy of KNeighborsClassifier     :",accuracy_score(y_test,y_pred2))
print("accuracy of Support Vector Machine   :",accuracy_score(y_test,y_pred3))
print("accuracy of GaussianNB               :",accuracy_score(y_test,y_pred4))
print("accuracy of RandomForestClassifier   :",accuracy_score(y_test,y_pred5))


# In[35]:


data = {'LR':accuracy_score(y_test,y_pred1), 'KNN':accuracy_score(y_test,y_pred2), 'SVC':accuracy_score(y_test,y_pred3),'GNB':accuracy_score(y_test,y_pred4),'RFC':accuracy_score(y_test,y_pred5)}
models=list(data.keys())
accuracy=list(data.values())
fig = plt.figure(figsize = (10, 5))
plt.bar(models, accuracy, color ='maroon', width = 0.3)
plt.xlabel("models")
plt.ylabel("accuracy")
plt.title("Accuracy of models ")
plt.show()


# In[39]:


dt={'age':50,'sex':1,'cp':0,'trestbps':128,'chol':146,'fbs':0,'restecg':1,'thalach':160,'exang':0,'oldpeak':2,'slope':0,'ca':1,'thal':2}
print(list(dt.values()))


# In[ ]:




