from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'

#Build KFold Project Object
# We’ll use 5 splits as that’s standard. Note that we want to create a single KFold object that all of the models will use
#It would be unfair if different models got a different split of the data.
kf = KFold(n_splits=5, shuffle=True)

#Now we’ll create three different feature matrices X1, X2 and X3. All will have the same target y.
X1 = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
X2 = df[['Pclass', 'male', 'Age']].values
X3 = df[['Fare', 'Age']].values
y = df['Survived'].values

#call function to see result of our features matrices
print("Logistic Regression with all features:")


def score_model(X, y, kf): 
    accuracy_scores = [] 
    precision_scores = [] 
    recall_scores = [] 
    f1_scores = [] 
    for train_index, test_index in kf.split(X): 
        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index] 
    model = LogisticRegression() 
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test) 
    accuracy_scores.append(accuracy_score(y_test, y_pred)) 
    precision_scores.append(precision_score(y_test, y_pred)) 
    recall_scores.append(recall_score(y_test, y_pred)) 
    f1_scores.append(f1_score(y_test, y_pred)) 
    print("accuracy:", np.mean(accuracy_scores)) 
    print("precision:", np.mean(precision_scores)) 
    print("recall:", np.mean(recall_scores)) 
    print("f1 score:", np.mean(f1_scores)) 

print("Logistic Regression with all fetures:")
score_model(X1,y,kf)
print()

print("Logistic Regression with Pclass, Sex, Age:")
score_model(X2,y,kf)
print()

print("Logistic Regression with Fare & Age features:")
score_model(X3,y,kf)
print()

#build with best model
model = LogisticRegression()
model.fit(X1, y)
prediction_test = model.predict([[3, False, 25, 0, 1, 2]])
print(prediction_test)