import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(path)

###CONTINUOS NUMERICAL VARIABLES
#List the data types of each column
#print(df.dtypes)

#Find the correlation between the following columns: bore, stroke, compression-ratio, and horsepower.
#print(df[['bore','stroke','compression-ratio','horsepower']].corr())

#Engine size as potential predictor variable of price
#sns.regplot(x="engine-size",y="price",data=df)
#plt.ylim(0,)

#Correlation btw 'engine-size' and 'price'
#print(df[['engine-size','price']].corr())

#Scatterplot of "highway-mpg" and "price"
#sns.regplot(x="highway-mpg",y="price",data=df)
#plt.ylim(0,)

#Correlation btw 'highway-mpg' and 'price'
#print(df[['highway-mpg','price']].corr())

#Scatterplot of "peak-rpm" and "price"
#sns.regplot(x="peak-rpm",y="price",data=df)
#plt.ylim(0,)

#Correlation btw 'peak-rpm' and 'price'
#print(df[['peak-rpm','price']].corr())


###CATEGORICAL VARIABLES

#Boxplot btw 'body-style' and 'price'
#sns.boxplot(x="body-style",y="price",data=df)

#Boxplot btw 'engine-location' and 'price'
#sns.boxplot(x="engine-location",y="price",data=df)

#Boxplot btw 'drive-wheel' and 'price'
#sns.boxplot(x="drive-wheels",y="price",data=df)


###DESCRIPTIVE STATISTICAL ANALYSIS 
#print(df.describe())    #This skips 'object' data types
#print(df.describe(include=['object']))  #This includes 'object'


###VALUE COUNTS
#print(df['drive-wheels'].value_counts())    #This is pandas series
drive_wheels_count = df['drive-wheels'].value_counts().to_frame()
drive_wheels_count.rename(columns={'drive-wheels':'value_counts'},inplace=True)
drive_wheels_count.index.name = 'drive-wheels'
#print(drive_wheels_count)

# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)
#print(engine_loc_counts)


###BASICS OF GROUPING 
#print(df['drive-wheels'].unique())

#df_group_one = df[['drive-wheels','body-style','price']]
#df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()

#df_group_two = df[['drive-wheels','body-style','price']]
#df_group_two = df_group_two.groupby(['body-style'],as_index=False).mean()

#print(df_group_one)
#rint(df_group_two)

df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
#print(grouped_test1)

grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot = grouped_pivot.fillna(0)     #Fill missing values with 0
#print(grouped_pivot)

#Use the "groupby" function to find the average "price" of each car based on "body-style".
#df_group_test4 = df[['price','body-style']]
#df_group_test4 = df_group_test4.groupby(['body-style'],as_index=False).mean()
#print(df_group_test4)


#Use a heat map to visualize the relationship btw 'body-style' versus 'price'
plt.pcolor(grouped_pivot,cmap='RdBu')
plt.colorbar()


fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')
#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

##insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)

#plt.show()


###CORRELATION AND CAUSATION 

#Calculate the Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'.
pearson_coef, p_value = stats.pearsonr(df['wheel-base'],df['price'])
#print("The Pearson Correlation Coefficient is",pearson_coef,"with a P-value of P =",p_value)

pearson_coef_1, p_value_1 = stats.pearsonr(df['horsepower'],df['price'])
#print("The Pearson Correlation Coefficient is",pearson_coef_1,"with a P-value of P =",p_value_1)


###ANOVA

#To see if different types of 'drive-wheels' impact 'price', we group the data.
grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
#print(grouped_test2.head(2))

#Obtain the same values for groupby method 
#grouped_test2.get_group('4wd')['price']

#f_score and p_value 
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'],grouped_test2.get_group('rwd')['price'],grouped_test2.get_group('4wd')['price'])
print("ANOVA results: F = ",f_val," P = ",p_val)
