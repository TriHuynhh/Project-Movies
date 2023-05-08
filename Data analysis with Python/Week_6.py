from asyncio.windows_utils import pipe
from operator import truediv
from re import T
from statistics import linear_regression
from turtle import screensize
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

file_name = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df = pd.read_csv(file_name)

df.drop(['id','Unnamed: 0'],axis=1,inplace=True)
#print(df.head(5))

print("Number of NaN values for the column bedrooms: ",df['bedrooms'].isnull().sum())
print("Number of NaN values for the column bathrooms: ",df['bathrooms'].isnull().sum())

mean_bedrooms = df['bedrooms'].mean()
mean_bathrooms = df['bathrooms'].mean()
df['bedrooms'].replace(np.nan,mean_bedrooms,inplace=True)
df['bathrooms'].replace(np.nan,mean_bathrooms,inplace=True)
print("Number of NaN values for the column bedrooms: ",df['bedrooms'].isnull().sum())
print("Number of NaN values for the column bedrooms: ",df['bathrooms'].isnull().sum())

print(df.columns.tolist())      # print out all the header 
uniq_floor = df['floors'].value_counts()
print(uniq_floor)

#print(df[['waterfront','view']].value_counts())

# Use the function boxplot in the seaborn library to 
# determine whether houses with a waterfront view or 
# without a waterfront view have more price outliers.

#sns.boxplot(x="waterfront",y=df['price'],data=df)


# Use the function regplot in the seaborn library
# to determine if the feature sqft_above is 
# negatively or positively correlated with price.

#sns.regplot(x="sqft_above",y=df['price'],data=df)
#plt.ylim(0,)
#plt.show()
#plt.close()

### MODULE 4: Model development

# Question 5
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
score1 = lm.score(X,Y)
print(score1)

# Question 6: 
x = df[['sqft_living']]
y = df['price']
lm.fit(x,y)
score2 = lm.score(x,y)
print(score2)

# Question 7: 
features = ["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
u = df[features]
v = df['price']
lm.fit(u,v)
score_test = lm.score(u,v)
print("The score_test is:",score_test)

# Question 8: 
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
#print(type(Input))
Pipe = Pipeline(Input)
print(Pipe)
Pipe.fit(u,v)
score_pipe = Pipe.score(u,v)
#print(type(Pipe))
#print(type(x))
print("The type of 'features' is:",type(features))
print(score_pipe)

### MODULE 5: Module evaluation and refinement

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

print(type(x_test))
print(type(x_train))
print("number of test samples:", x_test.shape[0])   # [0] for number of rows and [1] for number of columns 
print("number of training samples:",x_train.shape[0])

# Question 9 

from sklearn.linear_model import Ridge

RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(x_train,y_train)
score_ridge = RidgeModel.score(x_test,y_test)
print(score_ridge)

# Question 10
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RidgeModel.fit(x_train_pr,y_train)
score_ridge_1 = RidgeModel.score(x_test_pr,y_test)
print(score_ridge_1)