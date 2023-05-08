import pandas as pd

file_name = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df = pd.read_csv(file_name)

# Question 1:
print(df.dtypes)

# Question 2: 
df.drop(["id","Unnamed: 0"],axis=1,inplace=True)
print(df.describe())