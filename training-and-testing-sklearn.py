import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


#dataset before
df = pd.read_csv("https://sololearn.com/uploads/files/titanic.csv")
df['male'] = df['Sex'] == 'male'
X = df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
y = df['Survived'].values


#split training data
X_train, X_test, y_train, y_test = train_test_split(X,y)

# split with train size (example size 60% training data, 40% test data)
# train_test_split(X, y, train_size=0.6) 
print("Whole dataset: ", X.shape, y.shape)
print("Training set: ", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)

#build model with training dataset
model = LogisticRegression()
model.fit(X_train,y_train)
scoring_model = model.score(X_test,y_test)
print(scoring_model)

#build metric with new dataset
y_predict = model.predict(X_test)
accuracy_model = accuracy_score(y_test,y_predict)
precision_model = precision_score(y_test,y_predict)
recall_model = recall_score(y_test,y_predict)
f1_model_score = f1_score(y_test,y_predict)
print("Accuracy of model: ", accuracy_model)
print("Precision of model: ", precision_model)
print("Recall of model: ", recall_model)
print("F1 Score of model: ", f1_model_score)
