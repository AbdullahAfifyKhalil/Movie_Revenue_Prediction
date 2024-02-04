#!/usr/bin/env python
# coding: utf-8

# In[1629]:


import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


# In[1630]:


movie_data=pd.read_csv(r"tmdb-movies.csv") 


# In[1631]:


movie_data.head()


# In[1632]:


movie_data.info()


# In[1633]:


movie_data.isnull().sum()


# In[1634]:


#removing  row duplicates from the data frame.
movie_data.drop_duplicates(inplace = True)
movie_data.shape


# In[1635]:


print("Rows with zero values in the budget_adj column:",movie_data[(movie_data['budget_adj']==0)].shape[0])
print("Rows with zero values in the revenue_adj column:",movie_data[(movie_data['revenue_adj']==0)].shape[0])


# In[1636]:


#Replace un necessary rows with the mean value of each column
movie_data['budget_adj'] = movie_data['budget_adj'].replace(0, movie_data['budget_adj'].mean())  
movie_data['revenue_adj'] = movie_data['revenue_adj'].replace(0, movie_data['revenue_adj'].mean())     


# In[1637]:


sr=pd.Series(['production_companies','tagline']).is_unique
print(sr)


# In[1638]:


#Removing un necessary columns from the data frame
movie_data.drop(['id','imdb_id','homepage','director','tagline','release_date','production_companies'],axis=1,inplace=True)
movie_data.shape


# In[1639]:


movie_data.isnull().sum()


# In[1640]:


movie_data.shape


# In[1641]:


#Dealing with NaN values with proper imputation techniques 

movie_data['overview'].fillna(movie_data['overview'].mode()[0], inplace=True)

movie_data['cast'].fillna(movie_data['cast'].mode()[0], inplace=True)

movie_data['keywords'].fillna(movie_data['keywords'].mode()[0], inplace=True)

movie_data['genres'].fillna(movie_data['genres'].mode()[0], inplace=True)


# In[1642]:


movie_data.isnull().sum()


# In[1643]:


movie_data.head()


# In[1644]:


#Convert the used categorical columns to numerical columns using label encoding techniques 
label_encoder = preprocessing.LabelEncoder()
movie_data['original_title']= label_encoder.fit_transform(movie_data['original_title']) 
movie_data['cast']= label_encoder.fit_transform(movie_data['cast']) 
movie_data['overview']= label_encoder.fit_transform(movie_data['overview']) 
movie_data['keywords']= label_encoder.fit_transform(movie_data['keywords']) 


# In[1645]:


##Convert the used categorical columns to numerical columns using One hot encoding 
movie_data['genres']=movie_data['genres'].str.get_dummies(sep='|')
movie_data.head()


# In[ ]:





# In[1646]:


#the net profit which is the difference between (revenue_adj – budget_adj).
movie_data['netprofit']=movie_data['revenue_adj']-movie_data['budget_adj']
movie_data.head()


# In[1647]:


X = movie_data.drop(['budget_adj','revenue_adj','netprofit'],axis=1)
Y = movie_data['netprofit']


# In[1648]:


print(movie_data[(movie_data['netprofit']==0)].shape[0])


# In[1649]:


#movie_data['netprofit'] = movie_data['netprofit'].replace(0, movie_data['netprofit'].mean())  


# In[1650]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
X.head()


# In[1651]:


#movie_data=movie_data.astype(int)


# In[1652]:


movie_data.info()


# In[1653]:


#apply feature scaling (normalization) for variables 
scaler=StandardScaler()
X=scaler.fit_transform(X)


# In[1654]:


#Aplly Linear Regression Model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train,Y_train)


# In[1655]:


training_data_prediction = lin_reg_model.predict(X_train)
error_score = mean_squared_error(Y_train, training_data_prediction)
print("Mean squared training Error : ", error_score)


# In[1656]:


test_data_prediction = lin_reg_model.predict(X_test)
error_score = mean_squared_error(Y_test, test_data_prediction)
print("Mean squared testing Error : ", error_score)


# In[1657]:


plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual revenue")
plt.ylabel("Predicted revenue")
plt.title(" Actual revenue vs Predicted revenue")
plt.show()


# In[1658]:


plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual revenue")
plt.ylabel("Predicted revenue")
plt.title(" Actual revenue vs Predicted revenue")
plt.show()


# In[1659]:


#Apply ensemble methods to improve the accuracy of the model
GBRModel = GradientBoostingRegressor(n_estimators=100,max_depth=5,learning_rate = 1.5 ,random_state=33)
GBRModel.fit(X_train, Y_train)


# In[1660]:


print('GBRModel Train Score is : ' , GBRModel.score(X_train, Y_train))


# In[1661]:


print('GBRModel Test Score is : ' , GBRModel.score(X_test, Y_test))


# In[1662]:


y_train_pred = GBRModel.predict(X_train)
y_test_pred = GBRModel.predict(X_test)


# In[1663]:


MSEValue = mean_squared_error(Y_train, y_train_pred)
print('Mean Squared Train Error Value is : ', MSEValue)


# In[1664]:



MSEValue = mean_squared_error(Y_test, y_pred)
print('Mean Squared  Test Error Value is : ', MSEValue)


# In[1665]:


plt.scatter(Y_train, y_train_pred)
plt.xlabel("Actual revenue")
plt.ylabel("Predicted revenue")
plt.title(" Actual revenue vs Predicted revenue")
plt.show()


# In[1666]:


plt.scatter(Y_test, y_pred)
plt.xlabel("Actual revenue")
plt.ylabel("Predicted revenue")
plt.title(" Actual revenue vs Predicted revenue")
plt.show()

