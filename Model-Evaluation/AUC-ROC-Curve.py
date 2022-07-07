import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

#importing dataset
df = pd.read_csv("https://sololearn.com/uploads/files/titanic.csv")
df['male'] = df['Sex'] == 'male'
X = df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
y = df['Survived'].values

#split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y)

#training 1 
model1 = LogisticRegression()
model1.fit(X_train,y_train)
y_predict_proba = model1.predict_proba(X_test)
print("Model 1 Area Under Curve Score: ",roc_auc_score(y_test,y_predict_proba[:,1]))

#training 2
model2 = LogisticRegression()
model2.fit(X_train[:,0:2],y_train)
y_predict_proba2 = model2.predict_proba(X_test[:,0:2])
print("Model 2 Area Under Curve Score: ",roc_auc_score(y_test,y_predict_proba2[:,1]))
