from audioop import lin2adpcm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm


path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'

df = pd.read_csv(path)
df.to_csv('module_5.auto.csv')


###FUNCTION FOR PLOTTING

def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()

def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    plt.close()

### PART 1: TRAINING AND TESTING

y_data = df['price']
x_data = df.drop('price', axis=1)       #df.drop: drop specified labels from rows or columns. axis=1: drop column

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.1,random_state=1)

print("Number of test sample: ",x_test.shape[0])
print("Nuber of training samples: ",x_train.shape[0])

x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data,y_data,test_size=0.4,random_state=0)
print("Number of test sample: ",x_test1.shape[0])
print("Nuber of training samples: ",x_train1.shape[0])

lre = LinearRegression()
lre.fit(x_train[['horsepower']],y_train)

R_squared = lre.score(x_test[['horsepower']],y_test)
print(R_squared)

R_squared2 = lre.score(x_train[['horsepower']],y_train)
print(R_squared2)

R_squared3 = lre.score(x_test1[['horsepower']],y_test1)
print(R_squared3)

# CROSS-VALIDATION SCORE

Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
print(Rcross)
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())

yhat = cross_val_predict(lre, x_data[['horsepower']], y_data,cv=4)
print(yhat[0:5])


### PART 2: OVERFITTING, UNDERFITTING, AND MODEL SELECTION 

lr = LinearRegression()
lr.fit(x_train[['horsepower','curb-weight','engine-size','highway-mpg']],y_train)
yhat_train = lr.predict(x_train[['horsepower','curb-weight','engine-size','highway-mpg']])
yhat_test = lr.predict(x_test[['horsepower','curb-weight','engine-size','highway-mpg']])

print(yhat_train[0:5])
print(yhat_test[0:5])

#Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
#DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)

#Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
#DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)


## OVERFITTING

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)

pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])

poly = LinearRegression()
poly.fit(x_train_pr, y_train)

Yhat = poly.predict(x_test_pr)
print(Yhat[0:5])

print("Predicted value: ",Yhat[0:4])
print("Actual value: ",y_test[0:4].values)
#PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly,pr)

#R^2 of training data
pr_r_squared = poly.score(x_train_pr,y_train)

#R^2 of test data
pr_r_squared2 = poly.score(x_test_pr,y_test)

print(pr_r_squared)
print(pr_r_squared2) #Negative means overfitting

Rsqu_test = []
order = [1,2,3,4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    lr.fit(x_train_pr,y_train)
    Rsqu_test.append(lr.score(x_test_pr,y_test))

#plt.plot(order, Rsqu_test)
#plt.xlabel('order')
#plt.ylabel('R^2')
#plt.title('R^2 Using Test Data')
#plt.text(3, 0.75, 'Maximum R^2 ')  
#plt.show()

def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train,y_test, poly, pr)

pr1 = PolynomialFeatures(degree=2)
x_train_pr1 = pr1.fit_transform(x_train[['horsepower','curb-weight','engine-size','highway-mpg']])
x_test_pr1 = pr1.fit_transform(x_test[['horsepower','curb-weight','engine-size','highway-mpg']])
print(x_train_pr1.shape)
poly1 = LinearRegression()
poly1.fit = (x_train_pr1,y_train)
Yhat_test1 = poly1.predict = (x_test_pr1)
Title='Distribution Plot of Predicted Value Using test Data vs Data Distribution of Test Data'
#DistributionPlot(y_test,Yhat_test1,"Actual Values (Test","Predicted Values (Test)",Title)


# PART 3: RIDGE REGRESSION

pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train[['horsepower','curb-weight','engine-size','highway-mpg']])
x_test_pr = pr.fit_transform(x_test[['horsepower','curb-weight','engine-size','highway-mpg']])

RidgeModel = Ridge(alpha=1)
RidgeModel.fit(x_train_pr,y_train)
yhat_ridge = RidgeModel.predict(x_test_pr1)

print('predicted: ',yhat_ridge[0:4])
print('test set: ',y_test[0:4].values)

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    
    #pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
#plt.show()
#plt.close()


# PART 4: GRID SEARCH

parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]
print(type(parameters1))

RR = Ridge()
grid1 = GridSearchCV(RR,parameters1,cv=4)   # cv: cross validation (default = 5)
grid1.fit(x_data[['horsepower','curb-weight','engine-size','highway-mpg']],y_data)
bestRR = grid1.best_estimator_
print(bestRR)
bestRR_score = bestRR.score(x_test[['horsepower','curb-weight','engine-size','highway-mpg']],y_test)
print(bestRR_score)