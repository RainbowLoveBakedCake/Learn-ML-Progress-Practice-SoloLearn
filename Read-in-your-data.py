import pandas as pd

pd.options.display.max_columns = 6
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
print(df.describe())
print(df.head())

#Manipulating data with Pandas
#Single Column
col = df['Fare']
print(col)

#Multiple Columns
small_df = df[['Age','Sex','Survived']]
print(small_df)

#Creating Column
df['male'] = df['Sex'] == 'male'
print(df.head())

#Converting from a Pandas Series to a Numpy Array
#get values from selected column
print(df['Fare'].values)

#get values from multiple columns
print(df[['Pclass','Fare','Age']].values)

#to look at the shape like row and columns
arr = df[['Pclass','Fare','Age']].values
print(arr.shape)

#Select from a Numpy array
arr = df[['Pclass','Fare','Age']].values
#single element from array
arr[0,1]
print(arr[0])
#select single column for example Age, can use this syntax
print(arr[:,2])

#Masking to select column with certain criteria
#Mask is boolean array
arr = df[['Pclass','Fare','Age']].values[:10]
mask = arr[:,2] < 18
print(arr[mask])
print(arr[arr[:,2]<18])

#Sum and Counting
#using mask
mask = arr[:,2] < 18
print(mask.sum())
print(arr[arr[:,2]<18].sum())