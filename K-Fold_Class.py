from sklearn.model_selection import KFold
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

df = pd.read_csv("https://sololearn.com/uploads/files/titanic.csv")
X = df[['Age','Fare']].values[:6]
y = df['Survived'].values[:6]

kf = KFold(n_splits=3,shuffle=True)
for train, test in kf.split(X):
    print(train,test)

#first split
splits = list(kf.split(X))
first_split = splits[0]
print(first_split)
# (array([0, 2, 3, 5]), array([1, 4]))

#The first array is the indices for the training set and the second is the indices for the test set. Let’s create these variables.
train_indices, test_indices = first_split
print("training set indices:", train_indices)
print("test set indices:", test_indices)
# training set indices: [0, 2, 3, 5]
# test set indices: [1, 4]

#Now we can create an X_train, y_train, X_test, and y_test based on these indices.
X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

#If we print each of these out, we’ll see that we have four of the datapoints in X_train and their target values in y_train. The remaining 2 datapoints are in X_test and their target values in y_test.
print("X_train")
print(X_train)
print("y_train", y_train)
print("X_test")
print(X_test)
print("y_test", y_test)

#build a model
model = LogisticRegression()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

# Loop Over All the Folds
# We have been doing one fold at a time, but really we want to loop over all the folds to get all the values. 
#  We will put the code from the previous part inside our for loop.
scores = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
print(scores)
# [0.75847, 0.83146, 0.85876, 0.76271, 0.74011]

# Since we have 5 folds, we get 5 accuracy values. Recall, to get a single final value, we need to take the mean of those values.
print(np.mean(scores))
# 0.79029

#train a final model again
final_model = LogisticRegression()
final_model.fit(X,y)
print(final_model)