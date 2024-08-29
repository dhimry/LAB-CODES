#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt


# In[157]:


df=pd.read_csv('heights.csv',index_col=None)
df


# In[169]:


y=np.array(df['daughter height'])
x=np.array(df['mother height'])
alpha=0.000015
#y=b0+b1x
b0,b1=0,0
iteration_error=[]
for iter in range(30000):
    for i,j in zip(x,y):
            ypredict=b0+b1*i
            yreal=j
            gradient=ypredict-yreal
            iteration_error.append(gradient)
            b0=b0-alpha*gradient
            b1=b1-alpha*gradient*i


print(b0,b1)
ypred=b0+b1*x
mse=((ypred-y)**2).sum()
print('MSE Gradiant Descent: ',mse)
print('RMSE Gradiant Descent: ',mse**(1/2))
print('Predict: daughter height -63 :' , (b0+b1*63))


# In[150]:


plt.scatter(x,y)
plt.plot(x,ypred,color='red')


# In[151]:


from sklearn.linear_model import LinearRegression
X = df[['mother height']].values
y = df['daughter height'].values
print(X,y)
model = LinearRegression()

model.fit(X, y)
Sb0 = model.intercept_
Sb1 = model.coef_[0]
print(Sb0,Sb1)
Y=Sb0+Sb1*y
print(Y)
# print('MSE Gradiant Descent: ',mse)
# print('RMSE Gradiant Descent: ',mse**(1/2))


# In[152]:


x_axis = np.arange(24)
y_axis = abs(np.array(iteration_error[:24]))
plt.plot(x_axis, y_axis)
plt.xlabel("Iterations")
plt.ylabel("Absolute Error")
plt.show()


# In[172]:


df=pd.read_csv('study.csv',index_col=None)
x=df['hours']
y=df['Pass']


# In[217]:


b0,b1=0,0
error=[]
alpha=0.01
def compute_log_loss(y_true, y_pred):
    epsilon = 1e-15  # To prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
def sigmoid(z):
    return 1/(1+np.exp(-z))
for _ in range(10000):
    for i,j in zip(x,y):
        z=b0+b1*i
        y_pred=sigmoid(z)
        gradient=y_pred-j
        b0=b0-alpha*gradient
        b1=b1-alpha*gradient*i
        error.append(compute_log_loss(j,z))
print(b0,b1)
print(error[-2:])
z=b0+b1*x
y_pred=sigmoid(z)
threshold=0.5
print(y_pred)
y_pred=[1 if i>threshold else 0 for i in y_pred]
print(y_pred)
print(f'If Study for 3.5 hours: {1 if (sigmoid(b0+b1*3.5)) > threshold else 0}')
print(f'If Study for 7 hours: {1 if (sigmoid(b0+b1*7)) > threshold else 0}')
accuracy=0
for i,j in zip(y,y_pred):
    if i==j:
        accuracy+=1
accuracy=accuracy/(len(y))
print('accuracy',accuracy*100,'%')
plt.plot(np.arange(24),error[:24])


# 

# In[218]:


df = pd.read_csv('./data.csv')
x = np.array(df.drop(columns=['y']))
y = np.array(df['y'])


# In[229]:


w = np.zeros(2)
b = 0
error = []
epochs = 5
alpha = 0.01
def compute_log_loss(y_true, y_pred):
    epsilon = 1e-15  # To prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
for epoch in range(epochs):
    err = 0
    for i in range(len(x)):
        z = np.dot(x[i],w) + b
        y_pred = 1/(1 + np.exp(-z))
        err = compute_log_loss(y[i],y_pred)
        error.append(err)
        w = w + alpha*np.dot(x[i].T, (y[i]-y_pred))
        b = b + alpha*(y[i]-y_pred)
print(w)
print(b)
z = np.dot(x,w) + b
y_pred = 1/(1 + np.exp(-z))
# y_pred = [1 if i > thresh else 0 for i in y_pred]
print(y_pred)
print(error[:5])


# In[228]:


r = range(1,epochs+1)
plt.plot(np.arange(20),error[:20])
plt.show()


# In[ ]:




