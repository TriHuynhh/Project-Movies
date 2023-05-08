# Import libraries

import pandas as pd
import seaborn as sns
import numpy as np
import re

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

# Adjust the configuration of the plots we'll create
matplotlib.rcParams['figure.figsize'] = (12,8) 

# Read in the data

df = pd.read_csv(r'C:/Users/Asus/Desktop/Homework and Assignments/Movie/movies.csv')
# print(df.head())
# print(df.columns)

# ========================================DATA CLEANING=============================================


# See if there's any missing data
    
# print(df['rating'])

# rating_counts = df['rating'].value_counts()
# print(rating_counts)

df.dropna(
    subset=['rating','released','writer','company'],
    axis=0,
    inplace=True)


# 'score','votes','writer','budget','gross','company','runtime'
#print(df[['score','votes','writer','budget','gross','company','runtime']].mean())
mean_score = df['score'].mean()
df['score'].replace(np.nan,mean_score,inplace=True) # inplace to replace its position
df['votes'].replace(np.nan,df['votes'].mean(),inplace=True) # replace without needed to find mean previously
for col in {'budget','gross','runtime'}:        # loop for replace 
    df[col].replace(np.nan,df[col].mean(),inplace=True)


# for col in df.columns:      
#     pct_missing = np.mean(df[col].isna())
#     print('{} - {}%'.format(col,pct_missing))

# print(df[['budget','gross','runtime']].isna().sum())

# Data types for our columns 
#print(df.dtypes)

# Change data type of column
df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')
#print(df.dtypes)

# Create corrected year column
df['yearcorrect'] = df['released'].str.extract(pat = '([0-9]{4})').astype(int)
#print(df['released'].str.extract(pat = '([0-9]{4})'))

# Sort value based on 'gross' descending
df = df.sort_values(by=['gross'], inplace=False,ascending=False)
#print(df)
pd.set_option('display.max_rows',None)
#print(df)


# Drop any duplicates 
#print(df['company'].drop_duplicates())


# ====================================FINDING CORRELATION===================================


# Hypothesis to gross
#     budget high correlation 
#     company high correlation

# Scatter plot between budget and gross

plt.scatter(x=df['budget'],y=df['gross'])
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for films')
plt.title('Budget vs. Gross Earnings')
# plt.show()

# print(df.head())

# Using seaborn to plot budget vs gross
sns.regplot(x='budget',
            y='gross',
            data=df, 
            scatter_kws={"color":"red"},
            line_kws={"color":"blue"})
# plt.show()


# Let's start looking at correlation 
# print(df.corr())    # Default correlation method: Pearson/ Besides: Kendall, Spearman
# print(df.corr(method='kendall'))
# print(df.corr(method='spearman'))


# High correlation btw budget and gross
# Visualize the correlation 
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix,annot=True)
# default annot = false (khong the hien gia tri correlation), true = danh so
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.title('Correlation Matrix for Numeric Features')
# plt.show()

# Looks at Company 
df_numerized = df.copy()
for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtypes == 'object'):  # Find any column that have data typ as object 
        df_numerized[col_name] = df_numerized[col_name].astype('category')  # Change the data type of that column to category
        df_numerized[col_name] = df_numerized[col_name].cat.codes   # Assigning each individual cateogry with a distinct number 

# print(df_numerized.head(5))
# print(df.head(5))

correlation_matrix_numerized = df_numerized.corr()
sns.heatmap(correlation_matrix_numerized,annot=True)
# default annot = false (khong the hien gia tri correlation), true = danh so
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.title('Correlation Matrix for Numeric Features')
# plt.show()

# Unstacking for better visualization 
corr_mat_num = df_numerized.corr()
corr_mat_num_unstacked = corr_mat_num.unstack()
# print(corr_mat_num_unstacked)

sorted_corr = corr_mat_num_unstacked.sort_values()
print(sorted_corr[(sorted_corr)>0.5])
# --> Company and gross do not have high correlation 
# --> However, vote and budget have high correlation 



