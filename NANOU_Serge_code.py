
# coding: utf-8

#                                           TEST PYTHON CODE

# Using packages

# In[12]:

get_ipython().magic('matplotlib inline')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# 1-Import data AND back vizualisation

# In[2]:

df = pd.read_csv('C:/Users/akase/Desktop/Dossier_test/OnlineNewsPopularity.csv')


# In[3]:

df.shape


# The first elements of dataframe

# In[4]:

df.head()


# The last elements of dataframe

# In[5]:

df.tail()


# In[6]:

df.columns


# In[7]:

df.describe()


# In[8]:

df.info()


# 2-Target shares exploratory

# In[9]:

df[' shares'].describe()


# In[10]:

df[' shares']


# In[13]:

plt.hist(df[' shares'])


# 3-Outliers dropping

# In[14]:

#drop outliers in the dataset
df_new = df[(df[' shares'] < 200000) | (df[' shares'] == 200000)]  
df_new.shape


# In[15]:

#histogramm of variable share
sns.distplot(df_new[' shares'])


# In[16]:


df_new_1 = df[(df[' shares'] < 50000) | (df[' shares'] == 50000)]  
df_new_1.shape


# In[17]:

sns.distplot(df_new_1[' shares'])


# In[18]:

df_new_2 = df[(df[' shares'] < 10000) | (df[' shares'] == 10000)]  
df_new_2.shape


# In[19]:

sns.distplot(df_new_2[' shares'])


# In[20]:

df_new_3 = df[(df[' shares'] < 8000) | (df[' shares'] == 8000)]  
df_new_3.shape
df_new_3[' shares'].describe()


# In[21]:

df_new_3.shape


# In[22]:

#histogramm of variable kw_avg_avg
sns.distplot(df_new_2[' kw_avg_avg'])


# In[23]:

#dropping outliers 
df_new_2_1 = df_new_2[(df[' kw_avg_avg'] < 8000) | (df[' kw_avg_avg'] == 8000)]  
df_new_2_1.shape


# In[24]:

sns.distplot(df_new_2_1[' kw_avg_avg'])


# 3-Correlations of variable shares and explication 

# In[25]:

#correlations of a target variable
correlations = df_new_2_1.corr()
correlations = correlations[" shares"].sort_values(ascending=False)
correlations


# In[26]:

#correlation matrix
corr_matrix = df_new_2_1.corr()
f, ax = plt.subplots(figsize=(16, 16))
sns.heatmap(corr_matrix, vmax=.8, square=True);


# In[27]:

#share correlation matrix
k = 12 # number of variables
col_s = corr_matrix.nlargest(k, ' shares')[' shares'].index
c_m = np.corrcoef(df[col_s].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(c_m, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12}, yticklabels=col_s.values, xticklabels=col_s.values)
plt.show()


# 4-Regression models

# In[28]:

X = df_new_2_1.drop(['url', ' shares'], 1)  
y = df_new_2_1[' shares'] 


# In[29]:



std_scale = preprocessing.StandardScaler().fit(X)
X_scale = std_scale.transform(X)


# In[30]:

# regression using statsmodels
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())


# 5-Features selection using random forest

# In[31]:

df_new_2_1 = df_new_2_1.drop(['url'], 1)


# In[32]:

# Labels are the values we want to predict
labels = np.array(df_new_2_1[' shares'])


# In[33]:

# Convert to numpy array
df_new_2_1 = np.array(df_new_2_1)


# In[34]:

# feature importance using random forest
rf = RandomForestRegressor(n_estimators=80, max_features='auto')
rf.fit(X, y)
ranking = np.argsort(-rf.feature_importances_)
f, ax = plt.subplots(figsize=(15, 12))
sns.barplot(x=rf.feature_importances_[ranking], y=X.columns.values[ranking], orient='h')
ax.set_xlabel("feature importance")
plt.tight_layout()
plt.show()


# In[35]:

# use the top 10 features only
X= X.iloc[:,ranking[:10]]
print(X)


# In[36]:

X.shape


# 6-Prediction target variable(share) using random forest

# In[38]:

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(df_new_2_1, labels, test_size = 0.3, random_state = 42)


# In[39]:

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# In[40]:

# Instantiate model with 500 decision trees
rf = RandomForestRegressor(n_estimators = 500, random_state = 22)
# Train the model on training data
rf.fit(train_features, train_labels);


# In[41]:

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
print(errors)


# In[42]:

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)


# In[43]:

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

