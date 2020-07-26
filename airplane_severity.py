#!/usr/bin/env python
# coding: utf-8

# ##  "Machine learning model for Airplane accident severity predicition"

# ### Importing all required libraries

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set(style="whitegrid")
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

#import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# In[2]:


#importing the dataset
dataset = pd.read_csv('AirplaneAccident.csv')


# In[3]:


#getting an intuition of the data
dataset.head()


# ### Splitting the data into training set and test set

# In[4]:


y = dataset['Severity']
x = dataset.drop(['Severity','Accident_ID'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


# ### Check for any missing data

# In[5]:


dataset.info()


# In[6]:


dataset.describe().T


# ### Visualizing the data distribution

# In[7]:


class_label = dataset['Severity'].value_counts()
total_points = len(dataset)
print("Points with class label -> 'Highly fatal and damaging' are = ",class_label.values[0]/total_points*
100,"%")
print("Points with class label -> 'Significant damage and serious injuries' are = ",class_label.values[1]/total_points*
100,"%")
print("Points with class label -> 'Minor damage and injuries' are = ",class_label.values[2]/total_points*
100,"%")
print("Points with class label -> 'Significant damage and fatalities' are = ",class_label.values[3]/total_points*
100,"%")
labels = ['Highly fatal and damaging','Significant damage and serious injuries','Minor damage and injuries','Significant damage and fatalities']
sizes = [30.490000000000002,27.29,25.27,16.950000000000003]
colors = ['yellowgreen', 'gold','orange','green']
plt.figure(figsize=(8,10))
plt.pie(sizes, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True)


# ### 1. Training a random model and using it as a baseline
# 
# #### 1.1 Training a dummy classifier

# In[8]:


dummy_clf = DummyClassifier(strategy="uniform") # uniform means that the model randomly assigns a class label given a quiery point.
dummy_clf.fit(x_train, y_train)


# #### 1.2 Evaluating performance of random model using log loss

# In[9]:


print("Log loss of random model on 'Training data' = ",metrics.log_loss(y_train,dummy_clf.predict_proba(x_train),labels=class_label.index))

print("Log loss of random model on 'Testing data' = ",metrics.log_loss(y_test,dummy_clf.predict_proba(x_test),labels=class_label.index))


# As the model that predicted class label randomly(random model). Any model should perform better than this i.e the log loss of the model should be less than the log loss of the random model.

# ### 2. Univarite analysis
# 
# Training a model(eg: Logistic Regression) with a single feature and analyzing whether the feature is important or not!.

# In[10]:


lg = LogisticRegression(solver='lbfgs',multi_class='auto')
params = {'C':[0.0001,0.001,0.01,0.1,10,100,1000]}
gs = GridSearchCV(lg,param_grid=params,scoring='neg_log_loss')


# ### 2.1 Analyzing the importance of 'safety score' feature
# 
# We need to find out the best hyperparameters to use

# In[11]:


gs.fit(x_train['Safety_Score'].values.reshape(-1,1),y_train)
gs.best_params_


# Training the Logistic regression classifier with best hyper parameters

# In[12]:


lg = LogisticRegression(C=0.001,solver='lbfgs',multi_class='auto')
lg.fit(x_train['Safety_Score'].values.reshape(-1,1),y_train)
print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Safety_Score'].values.reshape(-1,1))))
print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Safety_Score'].values.reshape(-1,1))))


# ### 2.2 Analyzing the importance of ' Total safety complaints' feature
# 
# Finding best hyperparameters to use.

# In[13]:


gs.fit(x_train['Total_Safety_Complaints'].values.reshape(-1,1),y_train)
gs.best_params_


# Training the Logistic regression classifier with best hyper parameters

# In[14]:


lg = LogisticRegression(C=0.0001,solver='lbfgs',multi_class='auto')
lg.fit(x_train['Total_Safety_Complaints'].values.reshape(-1,1),y_train)
print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Total_Safety_Complaints'].values.reshape(-1,1))))
print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Total_Safety_Complaints'].values.reshape(-1,1))))


# ### 2.3 Analyzing the importance of ' Control metric ' feature
# 
# Finding best hyperparameters to use.

# In[15]:


gs.fit(x_train['Control_Metric'].values.reshape(-1,1),y_train)
gs.best_params_


# Training the Logistic regression classifier with best hyper parameters

# In[16]:


lg = LogisticRegression(C=0.001,solver='lbfgs',multi_class='auto')
lg.fit(x_train['Control_Metric'].values.reshape(-1,1),y_train)
print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Control_Metric'].values.reshape(-1,1))))
print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Control_Metric'].values.reshape(-1,1))))


# ### 2.4 Analyzing the importance of ' Turbulence in gforces ' feature
# 
# Finding best hyperparameters to use.

# In[17]:


gs.fit(x_train['Turbulence_In_gforces'].values.reshape(-1,1),y_train)
gs.best_params_


# Training the Logistic regression classifier with best hyper parameters

# In[18]:


lg.fit(x_train['Turbulence_In_gforces'].values.reshape(-1,1),y_train)
print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Turbulence_In_gforces'].values.reshape(-1,1))))
print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Turbulence_In_gforces'].values.reshape(-1,1))))


# ### 2.5 Analyzing the importance of 'Cabin Temperature' feature
# 
# Finding best hyperparameters to use.

# In[19]:


gs.fit(x_train['Cabin_Temperature'].values.reshape(-1,1),y_train)
gs.best_params_


# Training the Logistic regression classifier with best hyper parameters

# In[20]:


lg = LogisticRegression(C=0.01,solver='lbfgs',multi_class='auto')
lg.fit(x_train['Cabin_Temperature'].values.reshape(-1,1),y_train)
print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Cabin_Temperature'].values.reshape(-1,1))))
print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Cabin_Temperature'].values.reshape(-1,1))))


# ### 2.6 Analyzing the importance of 'Accident type code' feature
# 
# Finding best hyperparameters to use.

# In[21]:


gs.fit(x_train['Accident_Type_Code'].values.reshape(-1,1),y_train)
gs.best_params_


# Training the Logistic regression classifier with best hyper parameters

# In[22]:


lg = LogisticRegression(C=0.01,solver='lbfgs',multi_class='auto')
lg.fit(x_train['Accident_Type_Code'].values.reshape(-1,1),y_train)
print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Accident_Type_Code'].values.reshape(-1,1))))
print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Accident_Type_Code'].values.reshape(-1,1))))


# ### 2.7 Analyzing the importance of ' Max Elevation ' feature
# 
# Finding best hyperparameters to use.

# In[23]:


gs.fit(x_train['Max_Elevation'].values.reshape(-1,1),y_train)
gs.best_params_


# Training the Logistic regression classifier with best hyper parameters

# In[24]:


lg = LogisticRegression(C=0.0001,solver='lbfgs',multi_class='auto')
lg.fit(x_train['Max_Elevation'].values.reshape(-1,1),y_train)
print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Max_Elevation'].values.reshape(-1,1))))
print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Max_Elevation'].values.reshape(-1,1))))


# ### 2.8 Analyzing the importance of ' Violations ' feature
# Finding best hyperparameters to use.

# In[25]:


gs.fit(x_train['Violations'].values.reshape(-1,1),y_train)
gs.best_params_


# Training the Logistic regression classifier with best hyper parameters

# In[26]:


lg = LogisticRegression(C=0.01,solver='lbfgs',multi_class='auto')
lg.fit(x_train['Violations'].values.reshape(-1,1),y_train)
print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Violations'].values.reshape(-1,1))))
print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Violations'].values.reshape(-1,1))))


# ### 2.9 Analyzing the importance of ' Adverse wather metric ' feature
# Finding best hyperparameters to use.

# In[27]:


gs.fit(x_train['Adverse_Weather_Metric'].values.reshape(-1,1),y_train)
gs.best_params_


# Training the Logistic regression classifier with best hyper parameters

# In[28]:


lg = LogisticRegression(C=10,solver='lbfgs',multi_class='auto')
lg.fit(x_train['Adverse_Weather_Metric'].values.reshape(-1,1),y_train)
print("log loss on train = ",metrics.log_loss(y_train,lg.predict_proba(x_train['Adverse_Weather_Metric'].values.reshape(-1,1))))
print("log loss on test = ",metrics.log_loss(y_test,lg.predict_proba(x_test['Adverse_Weather_Metric'].values.reshape(-1,1))))


# ### Conclusion
# 
# Features like 'Security score', 'Control metric', 'Accident type code' could alone reduce the log loss on the class labels. And these three features also had a stability among train,test,cv datasets.(i.e the log loss for these features was less relatively and same for train,test and cv sets.)

# ### Analyzing the correlation matrix of training data

# In[31]:


cor = dataset.corr()
plt.figure(figsize=(12,10))
sns.heatmap(cor,annot=True)


# <p> <b>From the dark boxes in the above matrix (the point of intersection of 2 parameters we get that: </b></p>
# <p>Apart from the than turbulence and control metric, there is also a strong negative corelation between Adverse weather metric and accident type code.</p>
# <p> We cannot simple remove a vaiable, because we don't know which feature drives an optimal result.</p>
# <p>The same goes for the features safety score and days since inspection.</p>

# ### Adding the 'Total_Problems' feature

# In[32]:


temp = x_train['Violations'] + x_train['Total_Safety_Complaints']
x_train['Total_Problems'] = temp

temp = x_test['Violations'] + x_test['Total_Safety_Complaints']
x_test['Total_Problems'] = temp


# ### Applying Random Forests

# In[33]:


clf = RandomForestClassifier()
params = {'max_features': np.arange(1, 11),'criterion' :['gini', 'entropy']}
best_model = GridSearchCV(clf, params,n_jobs=-1)
best_model.fit(x_train,y_train)


# In[34]:


best_model.best_params_


# In[35]:


rf = RandomForestClassifier(criterion='entropy',max_features=8,n_estimators=1000)
rf.fit(x_train,y_train)
print(rf.score(x_test,y_test))


# In[36]:


x = dataset.drop(['Severity','Accident_ID'],axis=1)
x['Total_Problems'] = dataset['Violations'] + dataset['Total_Safety_Complaints']


# In[37]:


rf.fit(x,y)


# ### Applying xgboost and finalizing the model

# In[38]:


xgb = XGBClassifier()
xgb.fit(x,y)


# ### Testing the model created

# In[43]:


#Creating Test dataset 
test = dataset.drop(['Severity'],axis=1)
test_data = dataset.drop(['Accident_ID','Severity'],axis=1)
test_data['Total_Problems'] = test_data['Violations'] + test_data['Total_Safety_Complaints']
test_data


# ### Testing random forest classifier

# In[44]:


y_pred = rf.predict(test_data)
y_pred


# In[45]:


pred = pd.DataFrame()
pred['Accident_ID'] = test['Accident_ID']
pred['Severity'] = y_pred

###Printing predictions
pred


# ### Testing XGBClassifier

# In[46]:


y_pred = xgb.predict(test_data)
y_pred


# In[47]:


pred = pd.DataFrame()
pred['Accident_ID'] = test['Accident_ID']
pred['Severity'] = y_pred

###Printing predictions
pred

