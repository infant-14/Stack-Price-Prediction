#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.dates as mdates


# In[4]:


df = pd.read_csv('Minor Project Data set (Stock Price Prediction).csv')


# In[5]:


df.head(5)


# In[6]:


df.dtypes


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


print("Null valus: ",df.isnull().values.sum())
print("Na valus: ",df.isna().values.any())


# In[10]:


from pandas.plotting import lag_plot
plt.figure()
lag_plot(df['Open'], lag=3)
plt.title('google stock - Autocorrelation plot with lag = 3')
plt.show()


# In[11]:


#plot close price
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Close Prices')
plt.plot(df['Close'])
plt.title('ARCH CAPITAL GROUP closing price')
plt.show()


# In[12]:


df.hist(bins=50, sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12,12))
plt.show()


# In[13]:


correlation=df.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation,annot=True)


# In[14]:


from statsmodels.tsa.stattools import adfuller


# In[15]:


test_result=adfuller(df['Close'])


# In[16]:


def adfuller_test(close):
  result=adfuller(close)
  labels=['ADF test statistics','p-value','lags used','no of observations used']
  for value,label in zip(result,labels):
    print(label+':'+str(value))
  if result[1]<=0.05:
    print('reject null hypo,stat')
  else:
    print("accept null hypo,not stat")


# In[17]:


adfuller_test(df['Close'])


# In[19]:


from statsmodels.graphics.tsaplots import plot_acf


# In[22]:


fig,(ax1, ax2)=plt.subplots(1,2,figsize=(16,4))

ax1.plot(df.Close)
ax1.set_title("Original")
plot_acf(df.Close,ax=ax2);


# In[23]:


diff1=df.Close.diff().dropna()

fig,(ax1, ax2)=plt.subplots(1,2,figsize=(16,5))

ax1.plot(diff1)
ax1.set_title("Difference once")
plot_acf(diff1,ax=ax2);


# In[24]:


diff2=df.Close.diff().diff().dropna()

fig,(ax1, ax2)=plt.subplots(1,2,figsize=(16,5))

ax1.plot(diff2)
ax1.set_title("Difference once")
plot_acf(diff2,ax=ax2);


# In[25]:


get_ipython().system('pip install pmdarima')


# In[26]:


from pmdarima.arima.utils import ndiffs
ndiffs(df.Close,test="adf")
#it show the no of diff required is 1


# In[27]:


adfuller_test(diff1.dropna())


# In[28]:


from statsmodels.graphics.tsaplots import plot_pacf,plot_acf

diff=df.Close.diff().dropna()

fig,(ax1, ax2)=plt.subplots(1,2,figsize=(16,5))

ax1.plot(diff)
ax1.set_title("Difference once")
ax2.set_ylim(0,1)
plot_acf(diff,ax=ax2);


# In[29]:


diff=df.Close.diff().dropna()

fig,(ax1, ax2)=plt.subplots(1,2,figsize=(16,5))

ax1.plot(diff)
ax1.set_title("Difference once")
ax2.set_ylim(0,1)
plot_acf(diff,ax=ax2);


# In[30]:


from pmdarima import auto_arima
#ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")


# In[31]:


train=df.Close.iloc[:-30]
test=df.Close.iloc[-30:]
print(train.shape,test.shape)


# In[32]:


fit=auto_arima(train,trace=True,suppress_warnings=True)
fit.summary()


# In[33]:


print(df.shape)


# In[34]:


from statsmodels.tsa.arima.model import ARIMA


# In[35]:


model=ARIMA(df['Close'],order=(1,1,0))
model=model.fit()


# In[36]:


print(model.summary())


# In[37]:


#make predictions
start=len(train)
end=len(train)+len(test)-1
pred=model.predict(start=start,end=end,type='levels')

pred.index=df.index[start:end+1]
print(pred)


# In[38]:


df.Close.tail(5)


# In[39]:


result = pd.DataFrame({'Actual': test, 'Predicted':pred})
print(result)


# In[40]:


pred.plot(legend=True)
test.plot(legend=True)


# In[41]:


#actual vs predicted
from statsmodels.graphics.tsaplots import plot_predict
plot_predict(model,start=1,end=1000,dynamic=False);


# In[42]:


from sklearn import metrics
print(metrics.mean_squared_error(test,pred))
print(metrics.mean_absolute_error(test,pred))


# In[43]:


plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(train, 'green', label='Train data')
plt.plot(test, 'blue', label='Test data')
plt.plot(pred, 'yellow', label='pred')
plt.legend()


# In[ ]:




