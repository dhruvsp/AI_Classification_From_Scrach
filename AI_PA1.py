#!/usr/bin/env python
# coding: utf-8

# # EAS 595: Fundamental of Artificial Intellegence

# # Dhruv S. Patel(#50321707)

# # Project#1

# # --------------------------------------------------------------------------------------------------------------

# # Task 2

# ### Importing Library

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix


# ### Uploading CSV File 

# In[2]:


data=pd.read_csv("data.csv")
data.head()


# ### Converting M and B into 1 and 0 respectively for ease of computer understanding

# In[3]:


data=data.replace(to_replace="M",value=1)
data=data.replace(to_replace="B",value=0)
data.head()


# ### Dropping First column(ID)

# In[4]:


data_new=data.drop(['id'], axis = 1, inplace=True)
data.head()


# ### Defining Input and target

# In[5]:


x = data.drop(['diagnosis'],axis = 1)
y = data['diagnosis']


# ### Scaling the data

# In[6]:


from sklearn import preprocessing
x_scaled = preprocessing.scale(x)


# ### Splitting the data(80% for training and 20% for testing)

# In[28]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,train_size=0.8,random_state=2) #80% Test data and 20% Training Data
print(x_train.shape)
y_train=np.asmatrix(y_train).T
print(y_train.shape)


# ### Training Model

# In[8]:


LogisticRegr = LogisticRegression()
LogisticRegr.fit(x_train,y_train)
print (LogisticRegr.score(x_train, y_train))


# ### Report

# In[9]:


y_pred_test = LogisticRegr.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_test))


# ### Accuracy

# In[42]:


from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test,y_pred_test))


# ### Confusion Matrix

# In[11]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred_test)


# # --------------------------------------------------------------------------------------------------------------

# # Task 3 

# ### Initializing Parameter  

# In[12]:


epoch= 500 #Number of interation
learningrate = 0.06
w_initial=np.zeros((x_train.shape[1],1))
bias_initial=0


# ### Sigmoidal Function

# In[13]:


def sigmoid(z):
    return 1/(1+np.exp(-z))


# ### Genesys Equation

# In[35]:


def genesys(w,x,bias):
    y_cap=np.dot(x,w)+bias
    return sigmoid(y_cap)


# ### Loss function

# In[30]:


def lossFunction(w,x,y,bias):
    m=x.shape[0]
    y_pred=genesys(w,x,bias)
    error = np.multiply(y , np.log(y_pred)) +np.multiply((1-y),np.log(1-y_pred))
    cost = (-1/m) * error
    return np.sum(cost)


# ### Gradient Descent 

# In[40]:


def gradDescent(w,x,y,bias,learningRate):
    m=x_train.shape[0]
    y_pred=genesys(w,x,bias)
    diff=y-y_pred
    dw0=(-1/m)*np.sum(diff) #gradient of bias
    dw=(-1/m)*np.dot(x.transpose(),diff) #gradient of w
    bias=bias-learningRate*dw0
    w=w-learningRate*dw    
    return y_pred,bias,w


# In[34]:


def updateValues(w,x,y,bias,epochs,learning_rate):
    for i in range(epochs):
        loss=lossFunction(w,x,y,bias)
        y_pred,bias,w=gradDescent(w,x,y,bias,learning_rate)
    return w,bias


# In[36]:


w_updated,bias_updated=updateValues(w_initial,x_train,y_train,bias_initial,epoch,learningrate)


# ### Accuracy, Recall and Precision Calculation

# In[37]:


def accuracy_calc(y,y_pred):
    #True Positive --> Actual = 1, Predicted = 1
    #True Negative --> Actual = 0, Predicted = 0
    #False Positive--> Actual = 1, Predicted = 0
    #False Negative--> Actual = 0, Predicted = 1
    c_matrix=confusion_matrix(y, np.round(y_pred))
    tp=c_matrix[1][1]
    fn=c_matrix[1][0]
    fp=c_matrix[0][1]
    tn=c_matrix[0][0]
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    return c_matrix,accuracy,precision,recall


# In[38]:


y_test_pred=genesys(w_updated,x_test,bias_updated)
c_matrix,accuracy,precision,recall=accuracy_calc(y_test,y_test_pred)
print('-->Accuracy:',accuracy)
print('-->Precision:',precision)
print('-->Recall:',recall)


# In[41]:


print('-->Confusion Matrix')
print(np.transpose(c_matrix))


# In[ ]:




