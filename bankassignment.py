#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('https://raw.githubusercontent.com/manishanker/statistics_ML_jan_2020/master/bank.csv',sep=';')


# In[3]:


df.head()


# In[4]:


df['age'].median()


# In[5]:


df.isna().sum()/df.shape[0]


# In[6]:


df['job'].mode()


# In[7]:


df['job'].replace({'management':'employed','blue-collar':'employed','technician':'employed','admin.':'employed','services':'employed','retired':'unknown','self-employed':'employed','entrepreneur':'employed','housemaid':'employed','student':'unknown','unknown':'unknown','unemployed':'unemployed'},inplace=True)


# In[8]:


df['marital'].fillna('married',inplace=True)


# In[9]:


df['education'].fillna('unknown',inplace=True)


# In[10]:


df['education'].replace('na','unknown',inplace=True)


# In[11]:


df['balance'].replace('MANI',0,inplace=True)


# In[12]:


df['balance'].fillna(0,inplace=True)


# In[13]:


df['balance']=pd.to_numeric(df['balance'])


# In[14]:


df['housing'].fillna('yes',inplace=True)


# In[15]:


df['duration'].fillna(df['duration'].median(),inplace=True)


# In[16]:


df['y'].replace({'NO':'no','No':'no','yEs':'yes','Yes':'yes'},inplace=True)


# In[17]:


df['y'].fillna('no',inplace=True)


# In[18]:


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()


# In[19]:


df['default']=label.fit_transform(df['default'])
df['loan']=label.fit_transform(df['loan'])
# df['y']=label.fit_transform(df['y'])


# In[20]:


df['housing']=label.fit_transform(df['housing'])


# In[21]:


Maritial=pd.get_dummies(df['marital'],drop_first=True)
Maritial.head()


# In[22]:


Education=pd.get_dummies(df['education'])
Education


# In[23]:


df.head(1)


# In[24]:


Job=pd.get_dummies(df['job'])
Job


# In[25]:


Poutcome=pd.get_dummies(df['poutcome'])
Poutcome


# In[26]:


df=pd.concat([df,Job,Poutcome,Education,Maritial],axis=1)


# In[27]:


df.head()


# In[28]:


df.drop(['job','marital','education','contact','month','poutcome','unknown','day'],axis=1,inplace=True)


# In[29]:


df


# In[30]:


df.info()


# In[31]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
lgmodel=LogisticRegression()


# In[42]:


x=df.drop(['y'],axis=1)
y=df['y']


# In[43]:


x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.3,stratify=None)


# In[44]:


lgmodel=LogisticRegression()


# In[45]:


lgmodel=lgmodel.fit(x_train,y_train)


# In[41]:


x_train.shape


# In[39]:


y_train.shape


# In[50]:


x_train.isna().sum()


# In[55]:


y_train.isna().sum()


# In[ ]:




