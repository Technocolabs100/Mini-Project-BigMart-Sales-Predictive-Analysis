# Mini-Project-BigMart-Sales-Predictive-Analysis
#HYPOTHESIS
#The sales of products in Big Mart will increase by X% in the next quarter due to the increasing average temperature and the promotion of high-selling products during the summer season.
#LOADING PACKAGES AND DATA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# loading the data from csv file to Pandas DataFrame
big_mart_data = pd.read_csv('Train.csv')

print()
print("DATA STRUCTURE AND CONTENT")
big_mart_data.head()
print(big_mart_data.describe())
print()
print("MISSING VALUE IMPUTATION FOR FEATURE ENGINEERING")
missing_values_count = big_mart_data.isnull().sum()
missing_values_count
print()
# mean value of "Item_Weight" column
big_mart_data['Item_Weight'].mean()

# filling the missing values in "Item_weight column" with "Mean" value
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)

missing_values_count = big_mart_data.isnull().sum()
missing_values_count

print("mode of 'Outlet_Size' column")
big_mart_data['Outlet_Size'].mode()

print("filling the missing values in 'Outlet_Size' column with Mode")
mode_of_Outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))

print(mode_of_Outlet_size)

miss_values = big_mart_data['Outlet_Size'].isnull()

print(miss_values)
big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])
# checking for missing values
big_mart_data.isnull().sum()
print()
print()
print("DATA ANALYSIS")
big_mart_data.describe()
print()
print()
print("EXPLORATORY DATA ANALYSIS- it is an apporach to analyze data to summarize their main characteristics often with visual methods.")
sns.pairplot(big_mart_data)
print()
print()
print("***UNIVARIATE ANALYSIS***:- provides summary statistics for each field in the raw data set (or) summary only on one variable. Ex:- CDF,PDF,Box plot, Violin plot.")
sns.histplot(big_mart_data['Item_Visibility'])

sns.histplot(big_mart_data['Item_MRP'])

plt.figure(figsize=(10,5))
sns.histplot(big_mart_data['Item_Outlet_Sales'])

# Item_Type column
plt.figure(figsize=(20,6))
sns.countplot(x='Item_Type', data=big_mart_data)
plt.show()

# Item_Fat_Content column
plt.figure(figsize=(10,5))
sns.countplot(x='Item_Fat_Content', data=big_mart_data)
plt.show()

print("**BIVARIATE ANALYSIS**:- is performed to find the relationship between each variable in the dataset and the target variable of interest (or) using 2 variables and finding the relationship between them.Ex:-Box plot, Violin plot.")

big_mart_data.head()

sns.scatterplot(x=big_mart_data['Item_Weight'], y=big_mart_data['Item_MRP'])

sns.regplot(x=big_mart_data['Item_Weight'], y=big_mart_data['Item_MRP'])

#sns.swarmplot(data=big_mart_data, x="Item_Visibility", y="Item_Outlet_Sales")

print("**BOXPLOT TO SEE OUTLIERS IF ANY**")
big_mart_data.boxplot(by ='Item_Type', column =['Item_Outlet_Sales'], grid = False)

print("**ENCODING CATEGORICAL VALUES**")
pd.get_dummies(big_mart_data['Outlet_Size']).head()
pd.concat([big_mart_data['Outlet_Size'], pd.get_dummies(big_mart_data['Outlet_Size'])], axis=1).head()

print("#**LABEL ENCODING**")
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'species'.
big_mart_data['Item_Type']= label_encoder.fit_transform(big_mart_data['Item_Type'])
big_mart_data['Item_Type'].unique()

encoder = LabelEncoder()
big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])
big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])
#big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])
big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])
big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'])
big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])
big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])

big_mart_data.head()

print("**PREPROCESSING DATA**")
big_mart_data['Item_Fat_Content'].value_counts()
big_mart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)
big_mart_data['Item_Fat_Content'].value_counts()

print("**SPLITTING TARGET FROM FEATURES**")
print("**MODELLING**")
X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1) #ALL FEATURES EXCEPT ITEM_OUTLET_SALES
Y = big_mart_data['Item_Outlet_Sales'] #ONLU ITEM_OUTLET_SALES
print(X)
print(Y)


import matplotlib.pyplot as plt
from scipy import stats

print("**IMPLEMENTATION OF ORDINARY LEAST SQUARES REGRESSION USING STATSMODEL**")
import statsmodels.api as sm
#adding a constant
x = sm.add_constant(X)
#performing the regression
result = sm.OLS(Y, X).fit()
# Result of statsmodels 
print(result.summary())

print("**LINEAR REGRESSION**")
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, Y)
model = LinearRegression().fit(X, Y)

print("**R SQUARE FOR LINEAR REGRESSION MODEL**")
r_sq = model.score(X,Y)
print(f"coefficient of determination: {r_sq}")
plt.figure(figsize=(10,6))
sns.heatmap(big_mart_data.corr(),cmap=plt.cm.Reds,annot=True)
plt.title('Heatmap',fontsize=13)
plt.show()

print("we can see a strong negative correlation between \n 1.  Outlet_Location_Type and Outlet_Identifier \n 2.   outlet_size and outlet_location type \n 3.   outlet_size and outlet_type \n these are not continuous values. Therefore, we need to find other attributes which have a negative correlation. \n 1.   item_outlet_sales and item_visibility have a minute negative corelation")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

print()
print()

print("**RANDOM FOREST**")
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
#PRINT("RANDOM FOREST")
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X_train, Y_train)
melb_preds = forest_model.predict(X_test)
print("model validation")
print(mean_absolute_error(Y_test, melb_preds))
print()
print()



#**XGBOOST**
print("XGBOOST")
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)
# prediction on training data
training_data_prediction = regressor.predict(X_train)
# R squared Value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R Squared value = ', r2_train) #-------------------------------------------------------
# prediction on test data
test_data_prediction = regressor.predict(X_test)
# R squared Value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R Squared value = ', r2_test) #------------------------------------------------------------
test_data_prediction

print()
print()
print()

print("CONCLUSION")
print("I can sat that the Item Outlet Sales are strongly dependannt on the Item_MRP then by Outlet_type")
print("Increading the number of outlets of the type which are resulting in more items of the high MRP are required to increase the sales of the item and also the profit of the store in other areas also.")
