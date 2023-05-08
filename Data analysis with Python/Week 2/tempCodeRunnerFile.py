import pandas as pd
import numpy as np

filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df = pd.read_csv(filename,names=headers)

#Replace ? with NaN
df.replace("?",np.nan,inplace=True)

#Detect missing data 
missing_data = df.isnull()
#print(missing_data.head(5))
#for column in missing_data.columns.values.tolist():
#    print(column)
#    print(missing_data[column].value_counts())
#    print("")

#"normalized-losses": 41 missing data, replace them with mean
avg_norm_loss = df["normalized-losses"].astype(float).mean(axis=0)
df["normalized-losses"].replace(np.nan,avg_norm_loss,inplace=True)

#"stroke": 4 missing data, replace them with mean
ave_stroke = df["stroke"].astype(float).mean(axis=0)
df["stroke"].replace(np.nan,ave_stroke,inplace=True)

#"bore": 4 missing data, replace them with mean
ave_bore = df["bore"].astype(float).mean(axis=0)
df["bore"].replace(np.nan,ave_bore,inplace=True)

#"horsepower": 2 missing data, replace them with mean
ave_horse = df["horsepower"].astype(float).mean(axis=0)
df["horsepower"].replace(np.nan,ave_horse,inplace=True)

#"peak-rpm": 2 missing data, replace them with mean
ave_peakrpm = df["peak-rpm"].astype(float).mean(axis=0)
df["peak-rpm"].replace(np.nan,ave_peakrpm,inplace=True)

#"num-of-doors": 2 missing data, replace them with "four"
freq = df["num-of-doors"].value_counts().idxmax()
df["num-of-doors"].replace(np.nan,freq,inplace=True)

# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)

#print(df.head(5))

#Correct data format
#print(df.dtypes)

#print("")

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

#print(df.dtypes)


#DATA STANDARDIZATION

# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df["city-L/100km"] = 235/df["city-mpg"]

#Transform mpg to L/100km in the column of "highway-mpg" and change the name of column to "highway-L/100km".
#df["highway-L/100km"] = 235/df["highway-mpg"]

# transform mpg to L/100km by mathematical operation (235 divided by mpg)
df["highway-mpg"] = 235/df["highway-mpg"]

# rename column name from "highway-mpg" to "highway-L/100km"
df.rename(columns={'highway-mpg':'highway-L/100km'}, inplace=True)

# check your transformed data 
#print(df.head())


#DATA NORMALIZATION

# Replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()
#print(df[["length","width","height"]].head())


#BINNING

df["horsepower"] = df["horsepower"].astype('int',copy=True)

import matplotlib.pyplot as plt
#plt.hist(df["horsepower"])

# set x/y labels and plot title
#plt.xlabel("horsepower")
#plt.ylabel("count")
#plt.title("horsepower bins")

#3 bins of equal size bandwidth
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)

#Group names
group_names = ['Low', 'Medium', 'High']

#apply the function "cut" to determine what each value of df['horsepower'] belongs to
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)


#plt.bar(group_names, df['horsepower-binned'].value_counts())
# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")

#plt.show()

#BIN VISUALIZATION

#plt.hist(df['horsepower'],bins=3)
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")

#INDICATOR VARIABLE (OR DUMMY VARIABLE)




plt.show()




