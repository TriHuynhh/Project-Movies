# import pandas library
import pandas as pd
import numpy as np

#This function will download the dataset into your browser 

path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
#await download(path, "auto.csv")
#path="auto.csv"

df = pd.read_csv(path,header=None)

# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df.columns = headers

df1 = df.replace('?',np.NaN)
df=df1.dropna(subset=["price"], axis=0)
df.head(20)

#print(df.columns)
print(df.head(10))

df.to_csv("automobile.csv",index=False)

