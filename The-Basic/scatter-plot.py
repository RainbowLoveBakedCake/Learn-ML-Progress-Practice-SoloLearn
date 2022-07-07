import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.max_columns = 6
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
plt.scatter(df['Age'],df['Fare'])

#labeling
plt.xlabel('Age')
plt.ylabel('Fare')

#three scatter
plt.scatter(df['Age'],df['Fare'],c=df['Pclass'])

#Lining
plt.plot([0,80],[85,5])