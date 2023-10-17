#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#DIY0

import csv

f = open ('Lab01.csv')
count = 0
for row in csv.reader(f):
    if row != '\n':
        count += 1
print ('Number of rows:', count)


alist=[0]
for column in csv.reader(f):
    next(f)
    lab01[:10]=alist


import numpy as np

a=np.arange(5,16)
print(a)


b =np.linspace(0,23,7)
print(b)

c = np.random.uniform(low=-1, high=1, size=7)
print(c)

import matplotlib.pyplot as plt
d=np.sin(c)
plt.plot(c,d)

ra1 = np.random.randint(1,10,10)
ra2 = np.random.randint(11,20,10)
sub=ra2-ra1
dist=np.sqrt(sub)
print("Euclidean distance is ", dist)


# In[34]:


#DIY1


# In[35]:


import pandas as pd
passengerdata = pd.read_csv('PassengerData.csv')
passengerdata


# In[8]:


ticketprice = pd.read_excel('ticketPrices.xlsx')
ticketprice


# In[10]:


newdata= (pd.merge(passengerdata, ticketprice, on='TicketType'))
newdata


# In[22]:


e = newdata[newdata.Age >60]
e


# In[24]:


import matplotlib.pyplot as plt
import numpy as np

plt.scatter(newdata.Age, newdata.Fare)


# In[31]:


onex= newdata[(newdata.Sex == 'female') & ((newdata.Age>40) & (newdata.Age<50))]
onex


# In[32]:


oney= newdata[(newdata.Fare >=40)]
oney


# In[ ]:


plt.scatter(onex, oney)


# In[ ]:


#DIY2


# In[36]:


newdata2 = pd.read_csv('TitanicSurvivalData.csv')
newdata2


# In[40]:


print("Missing values in each column:\n", newdata2.isnull().sum()) 


# In[44]:


newdata2.Age.mean(axis = 0, skipna = True) 


# In[45]:


newdata2.Fare.mean(axis = 0, skipna = True) 


# In[54]:


newdata2["Age"].fillna("30", inplace = True)


# In[51]:


newdata2["Fare"].fillna("32", inplace = True)


# In[56]:


newdata2


# In[57]:


plt.scatter(newdata2.Age, newdata2.Fare)


# In[ ]:


#DIY3


# In[58]:


newdata3=pd.read_csv('TBReport.csv')


# In[59]:


newdata3


# In[60]:


print("Missing values in each column:\n", newdata3.isnull().sum()) 


# In[68]:


from matplotlib import pyplot as plt
import numpy as np
 
eee=newdata3.e_prev_100k_hi
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(eee)
plt.show()


# In[67]:


logeee=np.log(newdata3.e_prev_100k_hi)

fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(logeee)
plt.show()


# In[ ]:




