from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import pandas as pd

#dataset before
df = pd.read_csv("https://sololearn.com/uploads/files/titanic.csv")
df['male'] = df['Sex'] == 'male'
X = df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
y = df['Survived'].values
model = LogisticRegression()
model.fit(X,y)
y_predict = model.predict(X)


#metrics function
accuracy_model = accuracy_score(y,y_predict)
precision_model = precision_score(y,y_predict)
recall_model = recall_score(y,y_predict)
f1_model_score = f1_score(y,y_predict)
print("Accuracy of model: ", accuracy_model)
print("Precision of model: ", precision_model)
print("Recall of model: ", recall_model)
print("F1 Score of model: ", f1_model_score)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
table_confusion = confusion_matrix(y,y_predict)
print(table_confusion)
