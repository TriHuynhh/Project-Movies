import pandas as pd

file_name = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df = pd.read_csv(file_name)

# Question 1:
print(df.dtypes)

# Question 2: 
df.drop(["id","Unnamed: 0"],axis=1,inplace=True)
print(df.describe())

# Question 3: 
print(df['floors'].to_frame().value_counts())

# Question 4: 
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x="waterfront",y=df['price'],data=df)
plt.show()

# Question 5: 
sns.regplot(x="sqft_above",y=df['price'],data=df)
plt.ylim(0,)
plt.show()

# Question 6: 
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(df[['sqft_living']],df['price'])
score_q6 = lr.score(df[['sqft_living']],df['price'])
print("The R^2 of the question 6 is: ",score_q6)

# Question 7:
import numpy as np

mean_bedrooms = df['bedrooms'].mean()
mean_bathrooms = df['bathrooms'].mean()
df['bedrooms'].replace(np.nan,mean_bedrooms,inplace=True)
df['bathrooms'].replace(np.nan,mean_bathrooms,inplace=True)
list = ["floors","waterfront","lat","bedrooms","sqft_basement","view","bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
x = df[list]
y = df['price']
lr.fit(x,y)
score_q7 = lr.score(x,y)
print("The R^2 of the question 7 is: ",score_q7)

# Question 8: 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

input = [('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe = Pipeline(input)
pipe.fit(x,y)
score_q8 = pipe.score(x,y)
print("The R^2 of the question 8 is: ",score_q8)

# Question 9: 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

x_train, y_train, x_test, y_test = train_test_split()