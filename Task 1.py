#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


url = "http://bit.ly/w-data"
score_data = pd.read_csv(url)
score_data.head()


# In[9]:


score_data.tail()


# In[10]:


score_data.describe()


# In[11]:


score_data.shape


# In[12]:


score_data.columns


# In[13]:


score_data.info()


# In[14]:


score_data.isnull()


# In[15]:


plt.scatter(score_data['Hours'],score_data['Scores'])
plt.title('Hours vs Percentage')  
plt.xlabel('Hours')  
plt.ylabel('Scores')  
plt.show()


# In[17]:


x = score_data[['Hours']]
y = score_data.drop('Hours',axis = 1)

from sklearn.model_selection 
import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) ing(dividing ratio-80:20)

from sklearn.model_selection 
import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x_train.shape,x_test.shape


# In[8]:


from sklearn.linear_model 
import LinearRegression  

#instantiate the model
model = LinearRegression()

#fit the model
model.fit(x_train,y_train)
LinearRegression()

# Plotting the regression line
line = model.coef_*x+model.intercept_

# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line);
plt.show()


# In[15]:


#prediction using model
y_train_prediction = model.predict(x_train)
y_test_prediction = model.predict(x_test)

#calculating accuracy ofa model
acc_train = model.score(x_train,y_train)
acc_test = model.score(x_test,y_test)
from sklearn.metrics import mean_squared_error

#calculating Root Mean Squared Error
rms_train = np.sqrt(mean_squared_error(y_train,y_train_prediction))
rms_test = np.sqrt(mean_squared_error(y_test,y_test_prediction))
print("LINEAR REGRESSION MODEL: ")
print("\nAccuracy of Training Dataset: ",acc_train)
print("\nAccuracy of Testing Dataset: ",acc_test)
print("\nRMSE of Training Dataset: ",rms_train)
print("\nRMSE of Testing Dataset: ",rms_test)


# In[ ]:


#checking actual and predicted value
predicted = y_test_prediction[:,0]
actual = np.array(y_test['Scores'])
df_dict = {"Actual":actual,"Predicted":predicted}
df = pd.DataFrame(df_dict)
df.head()


# In[23]:


#plotting results for training data
plt.scatter(x_train,y_train,color = "cyan")
plt.plot(x_train,y_train_prediction,color = "black")
plt.grid(True)
plt.xlabel("Hours")
plt.ylabel("Score")
plt.title("(Hours vs Score)")


# In[7]:


#plotting results for testing data
plt.scatter(x_test,y_test,color = "red")
plt.plot(x_test,y_test_prediction,color = "black")
plt.grid(True)
plt.xlabel("Hours")
plt.ylabel("Score")
plt.title("(Hours vs Score)")


# In[19]:


#prediction of score of students if he studies for 9.25 hrs/day

output = model.predict([[9.25]])
print(f"The Score of student is {output[0][0]} if he studies for 9.25 hrs/day")


# In[20]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_prediction))


# In[ ]:




