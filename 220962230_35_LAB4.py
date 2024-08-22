#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[57]:


df=pd.read_csv('/home/ustudent/Documents/220962230/LAB4/experiment.csv')
df=df.drop(['Unnamed: 0'],axis=1)
n=len(df)


# Linear Regression

# In[60]:


X=np.array(df['temp'])
Y=np.array(df['yield'])
A=np.array([n , X.sum() , X.sum() , (X**2).sum()]).reshape(2,2)
B=np.array([Y.sum() , (Y*X).sum()]).reshape(2,1)
x=np.linalg.solve(A,B)
print(x)
b0=x[0][0]
b1=x[1][0]
print(b0,b1)
finaly=[]
for i in X:
    finaly.append(b0 + b1*i)
# plt.scatter(X,Y)
# plt.plot(X,finaly,color='red')
# plt.show()


# In[64]:


x_mean=X.sum()/n
y_mean=Y.sum()/n
sum1,sum2=0,0
for x,y in zip(X,Y):
    sum1+=(x-x_mean)*(y-y_mean)
    sum2+=(x-x_mean)**2
b1=sum1/sum2
b0=y_mean-b1*x_mean
print(b0,b1)


# Polynomial Regression

# In[95]:


B=np.array([Y.sum() , (Y*X).sum() , (Y*(X**2)).sum()]).reshape(3,1)
A=np.array([n,X.sum(),(X**2).sum()\
            , X.sum() ,(X**2).sum(),(X**3).sum(),\
            (X**2).sum(),(X**3).sum(),(X**4).sum()
           ]).reshape(3,3)
# print(A)
polyx=np.linalg.solve(A,B)
polyx=polyx.ravel()
print(polyx)
a0,a1,a2=polyx[0],polyx[1],polyx[2]
polyy=a0+a1*X+a2*(X*X)
# print(polyy)
plt.scatter(X,Y)
plt.plot(X,finaly,color='red')
plt.plot(X,polyy,color='green')


# In[93]:


mse=np.sum((Y-finaly)**2)/n
print(f'linear Regression : Mse: {mse} , Rmse : {mse**(1/2)}')
mse=np.sum((Y-polyy)**2)/n
print(f'Polynomial Regression : Mse: {mse} , Rmse : {mse**(1/2)}')

QUE2
# In[96]:


df=pd.read_csv('/home/ustudent/Documents/220962230/LAB4/heart.csv')
df


# In[104]:


x1=df['Area']
x2=df['X2']
x3=df['X3']
y=df['Infarc']
n=len(df)
A=np.array([n  ,  x1.sum()  , x2.sum()   , x3.sum()  ,\
           x1.sum() , (x1*x1).sum()  , (x1*x2).sum()  , (x1*x3).sum(),\
           x2.sum() , (x1*x2).sum()  ,(x2*x2).sum()   , (x2*x3).sum(),\
            x3.sum() , (x1*x3).sum() , (x2*x3).sum()  , (x3*x3).sum()
           ]).reshape(4,4)
B= np.array([y.sum() , (x1*y).sum() , (x2*y).sum() , (x3*y).sum()])
X=np.linalg.solve(A,B).ravel()
print(X)
b0,b1,b2,b3=X[0],X[1],X[2],X[3]
ynew= b0 + b1*x1 + b2*x2 + b3*x3
mse = ((y-ynew)**2).sum()/n
print(mse)
print(mse**(1/2))


# In[106]:


newdf=pd.DataFrame({'origional' : y , 'Predicted' : ynew})
print(newdf)


# In[ ]:




