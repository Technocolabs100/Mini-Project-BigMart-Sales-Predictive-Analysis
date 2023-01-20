# Mini-Project-BigMart-Sales-Predictive-Analysis
#importing  modules for project
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split

'''Project Description
The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined.
The aim of this data science project is to build a predictive model and find out the sales of each product at a particular store.Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.
The data has missing values as some stores do not report all the data due to technical glitches. Hence, it will be required to treat them accordingly.
Steps - Followed
Problem Statement
Hypothesis Generation
Loading Packages and Data
Data Structure and Content
Exploratory Data Analysis
Univariate Analysis
Bivariate Analysis
Missing Value Treatment
Feature Engineering
Encoding Categorical Variables
Label Encoding
One Hot Encoding
PreProcessing Data
Modeling
Linear Regression
Regularized Linear Regression
RandomForest
XGBoost
Summary'''

#1).Problem Statement

""" The aim of this data science project is to build a predictive model and 
    find out the sales of each product at a particular store"""
''''''
#2)Loading Packages and Data
pd.set_option("display.max_column",200)
test_data=pd.read_csv("F:\Scientist\MinI project\Test.csv")
print(test_data)

train_data=pd.read_csv("F:\Scientist\MinI project\Train.csv")
print(train_data)

concting=pd.concat([test_data,train_data])
print(concting)

#Exploratory Data Analysis

'''will be on train datsets'''

train_data=pd.read_csv("F:\Scientist\MinI project\Train.csv")
print(train_data)


print(train_data.value_counts())


#removing duplicated data
print(sum(train_data.duplicated()))

#again checking null values
print(train_data.isnull().sum())

duplicate=train_data.duplicated()
print(duplicate.sum())

                                        #Outlier Treatment

print(train_data.head())
'''
train_data.boxplot(column=['Item_Weight'])
sns.violinplot(column=['Item_MRP'],data=train_data)
plt.show()
'''

train_data.boxplot(column=['Item_Weight'])
plt.show()
train_data.boxplot(column=['Item_MRP'])
plt.show()
train_data.boxplot(column=['Item_Outlet_Sales'])
plt.show()

#here is Item_Outlet_Sales have many outliers
def remove_outlier(col):
    sorted(col)
    q1,q3=col.quantile([0.25,0.75])
    lower_range=q1-(1.5 * 0.25)
    upper_range=q3-(1.5 *0.75)
    return lower_range,upper_range
lowsales,uppsales=remove_outlier(concting['Item_Outlet_Sales'])
train_data['Item_Outlet_Sales']=np.where(train_data['Item_Outlet_Sales']>uppsales,uppsales,train_data['Item_Outlet_Sales'])
train_data['Item_Outlet_Sales']=np.where(train_data['Item_Outlet_Sales']<lowsales,lowsales,train_data['Item_Outlet_Sales'])
train_data.boxplot(column=['Item_Outlet_Sales'])
plt.show()

                                #handling missing values

#again checking null values
print(train_data.isnull().sum())


'''found that iteam weight and outlet size have null values'''
print(train_data['Item_Weight'].dtype)

'''placing mean values'''
train_data['Item_Weight']=train_data['Item_Weight'].fillna(value=train_data['Item_Weight'].mean())
print(train_data['Item_Weight'])
print(train_data['Item_Weight'].isnull().sum())

#outlet size
train_data['Outlet_Size']=train_data['Outlet_Size'].fillna('medium')

print(train_data['Outlet_Size'])
print(train_data['Outlet_Size'].isnull().sum())

# As Item_Visibility
# we can replace the '0' values with mean

train_data['Item_Visibility'] =train_data['Item_Visibility'].replace(0,train_data['Item_Visibility'].mean())
print(train_data['Item_Visibility'].value_counts())

                                        #9).Feature Engineering
# Here LF satnds for Low Fat and reg stands for Regular
print(train_data['Item_Fat_Content'].value_counts())
'''Low Fat    5089
Regular    2889
LF          316
reg         117
low fat     112
'''
# Replacing LF with Low Fat, reg Regular, low fat with Low Fat
train_data['Item_Fat_Content']= train_data['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})
print(train_data['Item_Fat_Content'].value_counts())


  #).Univariate Analysis

'''1. Calculate Summary Statistics'''
print(train_data.describe())
train_data_std=np.std(train_data)
print(f"\t test data std value \n ",train_data_std)


''' Create Frequency Table'''

print(f"\t value counts of each column \n ",train_data.value_counts())


'''. Create Charts'''

#ploting iteam types
sns.set_style('darkgrid')
sns.countplot(train_data['Item_Type'],order=train_data['Item_Type'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('iteam type')
plt.show()

sns.countplot(train_data['Item_Fat_Content'],order=train_data['Item_Fat_Content'].value_counts().index)
#plt.xticks(rotation=90)
plt.xlabel('Item_Fat_Content')
plt.show()

sns.countplot(train_data['Outlet_Type'],order=train_data['Outlet_Type'].value_counts().index)
#plt.xticks(rotation=90)
plt.xlabel('Outlet_Type')
plt.show()

sns.countplot(train_data['Outlet_Size'],order=train_data['Outlet_Size'].value_counts().index)
#plt.xticks(rotation=90)
plt.xlabel('Outlet_Size')
plt.show()

sns.kdeplot(train_data['Item_MRP'])
plt.show()

sns.jointplot(x=train_data['Item_Fat_Content'],y=train_data['Item_Type'],data=train_data)
sns.jointplot(x=train_data['Item_Type'],y=train_data['Item_Fat_Content'],data=train_data)
plt.xticks(rotation=90)
plt.show()

#comparing some datasets
x=train_data['Item_Fat_Content']
y=train_data['Item_Weight']
sns.set_style("whitegrid")
sns.displot(x="Item_Fat_Content",y="Item_Weight",data=train_data)
plt.show()

#checking price as per itms vias
topitems=train_data.groupby('Item_Type')['Item_MRP'].sum().sort_values(ascending=False).iloc[0:15]
plt.title('HIGHEST PRICE AS PER ITEAM')
plt.xlabel('ITEAM')
plt.ylabel('MRP')
topitems.plot(kind='bar')
plt.show()


                        #7).Bivariate Analysis
#finding corr and cov values\
covv=train_data.cov()
print("\n",covv)

corr=train_data.corr()
print("\n",corr)

sns.heatmap(train_data.corr(),annot=True)
plt.show()

                            #Encoding categorical to numerical
# Importing the required libary

from sklearn import preprocessing

print(train_data.head())
O_TYPE = pd.get_dummies(train_data['Outlet_Type'])
O_LOCATION_T = pd.get_dummies(train_data['Outlet_Location_Type'])
O_SIZE = pd.get_dummies(train_data['Outlet_Size'])
I_FAT = pd.get_dummies(train_data['Item_Fat_Content'])

NW_train_data = pd.concat([train_data,O_TYPE,O_LOCATION_T,O_SIZE,I_FAT],axis=1)
print(NW_train_data)

label = NW_train_data['Item_Outlet_Sales']

train = NW_train_data.drop(columns=['Item_Identifier','Item_Weight','Outlet_Type','Outlet_Identifier','Item_Fat_Content','Item_Outlet_Sales','Outlet_Location_Type','Item_Type','Outlet_Size','Outlet_Establishment_Year'])
print(train)
'''
data_types_dict = {'Item_Identifier': int,'Item_Fat_Content': int,'Item_Type': int,'Outlet_Identifier': int
    ,'Outlet_Size': int,'Outlet_Location_Type': int,'Outlet_Type': int}
df = train_data.astype(data_types_dict)
print(df)
   '''                                         #modling
'''
#importing required Ml library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
#tranfer data sets in training and testing
x=train.iloc[:,:-1].values
y=train.iloc[:,1].values
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.60,random_state=133)
print( x_train.shape,y_train.shape)
print( x_test.shape,y_test.shape)
#train the model
reqr=linear_model.LinearRegression()
reqr.fit(x_train,y_train)
scor=reqr.score(x_test,y_test)
print(scor)
#lassso re
LS=Lasso(alpha=0.05)
LS.fit(x_train,y_train)
v=LS.score(x_test,y_test)
print(v)
                #gradientBoostingRegressor
grad = GradientBoostingRegressor(n_estimators = 100)
grad.fit(x_train,y_train)
grad.score(x_test, y_test)
ran = RandomForestRegressor(n_estimators = 50)
ran.fit(x_train,y_train)
ran.score(x_test, y_test)
'''

from joblib import Parallel, delayed
import joblib

# Save the model as a pickle in a file
#joblib.dump(grad, 'mini model.pkl')

# Load the model from the file
l_joblib = joblib.load('mini model.pkl')
print(l_joblib)


                                    #prediction using Test data

test_data=pd.read_csv("F:\Scientist\MinI project\Test.csv")
print(test_data)
 #Encoding categorical to numerical
# Importing the required libary

from sklearn import preprocessing

print(test_data.head())
print(test_data.columns)

# Replacing LF with Low Fat, reg Regular, low fat with Low Fat
test_data['Item_Fat_Content']= test_data['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})
print(f"--------------------------------------------------------",test_data['Item_Fat_Content'].value_counts())


O_TYPE1 = pd.get_dummies(test_data['Outlet_Type'])
O_LOCATION_T1 = pd.get_dummies(test_data['Outlet_Location_Type'])
O_SIZE1 = pd.get_dummies(test_data['Outlet_Size'])
I_FAT1 = pd.get_dummies(test_data['Item_Fat_Content'])

df1 = pd.concat([test_data,O_TYPE1,O_LOCATION_T1,O_SIZE1,I_FAT1],axis=1)
test = df1.drop(columns=['Item_Identifier','Item_Weight','Outlet_Identifier','Item_Type','Outlet_Establishment_Year',
                         'Outlet_Size','Outlet_Type','Outlet_Location_Type','Item_Fat_Content'])
print(test)
#removing duplicated data
print(sum(test_data.duplicated()))

#again checking null values
print(test_data.isnull().sum())

duplicate=test_data.duplicated()
print(duplicate.sum())

test_data['Item_Weight']=test_data['Item_Weight'].fillna(value=test_data['Item_Weight'].mean())
print(test_data['Item_Weight'])
print(test_data['Item_Weight'].isnull().sum())

#outlet size
test_data['Outlet_Size']=test_data['Outlet_Size'].fillna('medium')

print(test_data['Outlet_Size'])
print(test_data['Outlet_Size'].isnull().sum())






                                  #modling

#importing required Ml library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso

#tranfer data sets in training and testing
x=train.iloc[:,:-1].values
y=train.iloc[:,1].values
print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.60,random_state=133)
print( x_train.shape,y_train.shape)
print( x_test.shape,y_test.shape)


#train the model

reqr=linear_model.LinearRegression()
reqr.fit(x_train,y_train)

scor=reqr.score(x_test,y_test)
print(f"\t linear regression prediction\n",scor)


#lassso re
LS=Lasso(alpha=0.05)
LS.fit(x_train,y_train)
v=LS.score(x_test,y_test)
print(v,f"\t lassso regression prediction\n")
 #gradientBoostingRegressor

grad = GradientBoostingRegressor(n_estimators = 100)
grad.fit(x_train,y_train)
g=grad.score(x_test, y_test)
print(g)

ran = RandomForestRegressor(n_estimators = 50)
ran.fit(x_train,y_train)
r=ran.score(x_test, y_test)
print(r,f"\t RandomForest regresssion regression prediction\n",)






# Load the model from the file

l_joblib = joblib.load('mini model.pkl')
print(l_joblib)

# Use the loaded model to make predictions
y_pred=l_joblib.predict(test)
print(y_pred)

pred=pd.DataFrame(y_pred,columns=["Sales"])
print(pred.head())

# Saving the predictions in csv file
print(pred.to_csv("Predictions.csv"))
