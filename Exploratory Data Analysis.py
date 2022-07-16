#!/usr/bin/env python
# coding: utf-8

# # Insurance Cost Prediction
# 
# **Research Question:**
# 
# Predicting the insurance cost of all the people based on various independent variabels available in the dataset from kaggle

# Importing necessary libraries for our program

# In[28]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


dataframe = pd.read_csv('insurance_data.csv')
dataframe


# **Meta Data:**
# 
# Rows: 1337
# 
# Features - 06
# 
# Target Variable: (charges)
# 
# 
# 1. Age: Numeric columns with integer values
# 2. sex: Categorical column with value either 'male' or 'female'
# 3. bmi: Body mass index, ratio of height to weight (kg / m ^ 2)
# 4. children: No. of children
# 5. smoker: Categorical column for smoking preference either 'yes' or 'no'
# 6. region: Region with the country. [The data set taken from kaggle corresponds to the people from US]
# 7. charges: Medical charges billed by health insurance (in $)

# In[4]:


dataframe.describe()


# In[5]:


dataframe.info()


# Here we can see, all the numeric columns [age, bmi, children, charges] with insights to their values.
# 
# 1. (Update) All the numberic values have equal number of counts which signifies no NaN or NA values
#    exists in the database. Just to make sure we will caculate them again below.
# 2. The maximum value in 'charges' is bit high as compared the 75% percentile. Which shows
#    medical charges can differ from individual based on their medical conditions.

# In[7]:


#Checking for NaN values
dataframe.isna().sum()

#Checking for Null values
# dataframe.isnull().sum().sum()


# In[11]:


#Checking for duplicate values if any
dataframe.duplicated().any() #.sum() will gives the count


# In[12]:


#There exist one or more duplicate values in the dataframe
dataframe.drop_duplicates(inplace=True)


# In[15]:


dataframe['smoker'].value_counts()


# In[26]:


dataframe.corr()


# From the above correlation matrix we can see there is no strong relation amoung the variables but a very weak relation

# In[25]:


sns.histplot(dataframe['charges'],bins=10, stat='count', kde=True)


# Visualizing the Dataset by ploting the Column values

# In[14]:


# Plotting 'smoker'
sns.pairplot(dataframe, hue='smoker', palette='cool', corner=True)
plt.show()


# From the above graphs we can clearly see how Non-Smokers are generally billed less as compared to the Smoker population.

# In[70]:


# Plotting 'region'
sns.pairplot(dataframe, hue='region', palette='cool')
plt.show()


# In[27]:


# plt.hist(dataframe[['charges']])
plt.scatter(dataframe[['age']], dataframe[['charges']], cmap='coolwarm', c=dataframe['bmi'])
plt.colorbar(label='BMI')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.show()


# It is pretty clear that with high BMI (Body Mass Index) the insurance cost is generally high for all the people irrespective of the ages, and with increase in age the insurance cost increases as well irrespective of any other condition.
# 

# In[ ]:




