from this import d
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("https://sololearn.com/uploads/files/titanic.csv")
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age','Siblings/Spouses',"Parents/Children",'Fare']].values
y = df['Survived'].values
print(X)
print(y)

# modeling with sklearn
model = LogisticRegression()
model.fit(X,y)
print(model.coef_, model.intercept_) 

#Make prediction with Model
predict1 = model.predict([[3, True, 22.0, 1, 0, 7.25]])
print(predict1)
predict2 = model.predict(X[:5])
print(predict2)
print(y[:5])

#Scoring model
model_score = model.score(X,y)
print(model_score)