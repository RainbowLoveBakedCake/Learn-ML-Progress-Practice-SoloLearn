import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support

sensitivity_score = recall_score
def specifity_score(y_true, y_predict):
    p, r, f, s = precision_recall_fscore_support(y_true,y_predict)
    return r[0]

#dataset before
df = pd.read_csv("https://sololearn.com/uploads/files/titanic.csv")
df['male'] = df['Sex'] == 'male'
X = df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
y = df['Survived'].values

#split data into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=5)
model = LogisticRegression()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)

#sensitivity and spesifity data
sensitivity_data_score = sensitivity_score(y_test, y_predict)
spesifity_data_score = specifity_score(y_test,y_predict)
print(sensitivity_data_score)
print(spesifity_data_score)

#Adjusting the Logistic Regression Threshold in Sklearn, default 0.5
y_model = model.predict_proba(X_test)[:,1] > 0.75
precision_model_score = precision_score(y_test, y_predict)
recall_model_score = recall_score(y_test,y_predict)
print(precision_model_score)
print(recall_model_score)
