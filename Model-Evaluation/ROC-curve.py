import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support, roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv("https://sololearn.com/uploads/files/titanic.csv")
df['male'] = df['Sex'] == 'male'
X = df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X,y)


model = LogisticRegression()
model.fit(X_train,y_train)
y_predict_proba = model.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test,y_predict_proba[:,1])

plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - specificity')
plt.ylabel('sensitivity')
plt.show()