#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
customer_churn=pd.read_csv("C:/Users/RAGHAVENDRA/Desktop/data sets from intellipaat/customer_churn.csv")


# In[3]:


customer_churn.head()


# # A) Solution

# In[4]:


customer_5=customer_churn.iloc[:,4]
customer_5.head()


# In[5]:


customer_15=customer_churn.iloc[:,14]
customer_15.head()


# In[6]:


senior_male_electronic=customer_churn[(customer_churn['gender']=="Male") & (customer_churn['SeniorCitizen']==1) & (customer_churn['PaymentMethod']=='Electronic check')]
senior_male_electronic.head()


# In[7]:


customer_total_tenure=customer_churn[(customer_churn['tenure']>70) | (customer_churn['MonthlyCharges']>100)]
customer_total_tenure.head()


# In[8]:


two_mail_yes=customer_churn[(customer_churn['Contract']=="Two year") & (customer_churn['Churn']=="Yes")]
two_mail_yes.head()


# In[9]:


customer_333=customer_churn.sample(333)
customer_333.head()


# In[12]:


customer_churn['Churn'].value_counts()


# # B) Solution: Data Vizualisation

# ### Bar plot

# In[10]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(customer_churn['InternetService'].value_counts().keys().tolist(),customer_churn['InternetService'].value_counts().tolist(),color="orange")
plt.title("Distribution of Internet Service")
plt.xlabel("Categories of Internet Service")
plt.ylabel("Count of Categories")
plt.show()


# # Histogram

# In[6]:


plt.hist(customer_churn['tenure'],bins=30,color="green")
plt.title("Distribution of tenure")
plt.show()


# # Scatter Plot

# In[20]:


plt.scatter(x=customer_churn['tenure'],y=customer_churn['MonthlyCharges'],color="brown")
plt.xlabel("Tenure of customer")
plt.ylabel("Monthly Charges of customer")
plt.title("Tenure vs Monthly Charges")

plt.show()


# # Box Plot

# In[7]:


customer_churn.boxplot(['tenure'],['Contract'],showmeans=True)


# # c) Solution: Linear Regression

# In[12]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

y=customer_churn[["MonthlyCharges"]]
x=customer_churn[["tenure"]]


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.3)


# In[14]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[15]:


regression=LinearRegression()
regression.fit(x_train,y_train)


# In[16]:


y_pred=regression.predict(x_test)


# In[17]:


from sklearn.metrics import mean_squared_error
import numpy as np


# In[23]:


print("Root Mean Squre: ",np.sqrt(mean_squared_error(y_pred,y_test)))


# In[ ]:





# # D) Solution

# ## Simple Logistic Regression

# In[2]:


x=pd.DataFrame(customer_churn.loc[:,"MonthlyCharges"])
y=pd.DataFrame(customer_churn.loc[:,"Churn"])


# In[3]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=1)


# In[4]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()


# In[5]:


logreg.fit(x_train,y_train)


# In[7]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[8]:


y_pred=logreg.predict(x_test)


# In[14]:


print("confusion matrix:\n\n",confusion_matrix(y_pred,y_test))


# In[15]:


print("Accuracy Score:\n\n",accuracy_score(y_pred,y_test))


# ## Multiple Logistic Regression

# In[16]:


x=pd.DataFrame(customer_churn.loc[:,["tenure",'MonthlyCharges']])
y=pd.DataFrame(customer_churn.loc[:,"Churn"])


# In[17]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=1)


# In[18]:


logreg=LogisticRegression()


# In[19]:


logreg.fit(x_train,y_train)


# In[21]:


y_pred=logreg.predict(x_test)


# In[26]:


print("Confusion matrix:\n\n\n",confusion_matrix(y_pred,y_test))
print("\n\nAccuracy score:",accuracy_score(y_pred,y_test))


# # E) Solution:Decison Tree

# In[ ]:





# In[29]:


y=pd.DataFrame(customer_churn.loc[:,"Churn"])
x=pd.DataFrame(customer_churn.loc[:,"tenure"])


# In[30]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=1)


# In[35]:


from sklearn.tree import DecisionTreeClassifier


# In[36]:


dec=DecisionTreeClassifier()
dec.fit(x_train,y_train)


# In[37]:


y_pred=dec.predict(x_test)


# In[38]:


print("Confusion matrix:\n\n\n",confusion_matrix(y_pred,y_test))
print("\n\nAccuracy score:",accuracy_score(y_pred,y_test))


# # F) Solution :RandomForest

# In[40]:


x=pd.DataFrame(customer_churn.loc[:,["tenure","MonthlyCharges"]])
y=pd.DataFrame(customer_churn.loc[:,"Churn"])


# In[41]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.3)


# In[46]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:





# In[48]:


ran=RandomForestClassifier(n_estimators=150)


# In[49]:


ran.fit(x_train,y_train)
y_pred=ran.predict(x_test)


# In[54]:


print("Confusion matrix:\n{} ".format(confusion_matrix(y_pred,y_test)))


# In[55]:


print("Accuracy_score:{}".format(accuracy_score(y_pred,y_test)))

