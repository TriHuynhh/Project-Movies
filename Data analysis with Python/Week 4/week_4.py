from statistics import linear_regression, mean
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(path)

#1. Linear regression and multiple linear regression

##Simple linear regression
lm = LinearRegression()

X = df[['highway-mpg']]
Y = df['price']

lm.fit(X,Y)

Yhat = lm.predict(X)
print(Yhat[0:5])
print(lm.intercept_)
print(lm.coef_)

lm1 = LinearRegression()

##Multiple linear regression

Z = df[['horsepower','curb-weight','engine-size','highway-mpg']]

lm.fit(Z,df['price'])

print(lm.intercept_)
print(lm.coef_)

lm2 = lm.fit(df[['normalized-losses','highway-mpg']],df['price'])
print(lm2.coef_)


#2. Model evaluation using visualization 

##Regression plot

#width = 12
#height = 10 


#sns.regplot(x="highway-mpg", y= "price", data = df)
#plt.ylim(0,)

#sns.regplot(x="peak-rpm",y="price",data=df)
#plt.ylim(0,)
#plt.show()

#print(df[["peak-rpm","highway-mpg","price"]].corr())

##Residual plot
width = 12
height = 10 
#plt.figure(figsize=(width,height))
#sns.residplot(x="highway-mpg",y="price",data=df)
#plt.show()


##MULTIPLE LINEAR REGRESSION
Z = df[['horsepower','curb-weight','engine-size','highway-mpg']]
print("The type of Z is:",type(Z))
lm.fit(Z, df['price'])
Y_hat = lm.predict(Z)
plt.figure(figsize=(width, height))

#ax1 = sns.distplot(df['price'],hist=False,color="r",label="Actual Value")
#sns.distplot(Y_hat,hist=False,color="b",label="Fitted Value",ax=ax1)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

#plt.show()
#plt.close()


##POLYNOMIAL REGRESSION AND PIPELINES

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

x = df['highway-mpg']
y = df['price']

f = np.polyfit(x, y, 3)
p = np.poly1d(f)
#print(f)
#print(p)

#PlotPolly(p,x,y,'highway-mpg')
#np.polyfit(x,y,11)

from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=2)
print(pr)

z_pr = pr.fit_transform(Z)
print(Z.shape)
print(z_pr.shape)

#PIPELINE

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Input = [('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe = Pipeline(Input)

print(pipe)

Z = Z.astype(float)
pipe.fit(Z,Y)
ypipe = pipe.predict(Z)
print(ypipe[0:4])

##Question: Create a pipeline that standardizes the data, then produce a prediction using a linear regression model using the features Z and target y.

input1 = [('scale',StandardScaler()),('model',LinearRegression())]
pipe1  = Pipeline(input1)
pipe1.fit(Z,Y)

ypipe1 = pipe1.predict(Z)
print(ypipe1[0:10])

#MEASURE FOR IN-SAMPLE EVALUATION

##MODEL 1: SIMPLE LINEAR REGRESSION
#Highway_mpg_fit

lm.fit(X,Y)
#Find the R^2 
print('The R-square is: ', lm.score(X,Y))

#Predict the output 
Yhat = lm.predict(X)
print('The output of the first 4 predicted value is: ', Yhat[0:4])

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(df['price'],Yhat)
print('The mean square error of price and predicted value is: ',mse)

##MODEL 2: MULTIPLE LINEAR REGRESSION 

lm.fit(Z, df['price'])
print('The R-square is: ', lm.score(Z, df['price']))

Y_predict_multifit = lm.predict(Z)

print('The mean square error of price and predicted value using multifit is: ',mean_squared_error(df['price'],Y_predict_multifit))

##MODEL 3: POLYNOMIAL FIT

from sklearn.metrics import r2_score

r_squared = r2_score(Y, p(x))
print('The R-square value is: ', r_squared)

##MSE

print(mean_squared_error(df['price'],p(x)))

##PREDICTION AND DECISION MAKING 

new_input = np.arange(1,100,1).reshape(-1,1)
#print(new_input)

lm.fit(X,Y)
print(lm)

yhat = lm.predict(new_input)
print(yhat[0:5])

plt.plot(new_input,yhat)
plt.show()