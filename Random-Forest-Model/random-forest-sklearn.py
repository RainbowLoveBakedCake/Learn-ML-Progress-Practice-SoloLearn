import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data['data'])
columns = cancer_data['feature_names']
df['target'] = cancer_data['target']

X = df[cancer_data.feature_names].values
y = df['target'].values
print('data dimension',X.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=101)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

first_row = X_test[0]
print("prediction:", rf.predict([first_row]))
print("true value:", y_test[0])

#prediction : [1]
#true value : 1

print("random forest accuracy:",rf.score(X_test,y_test))
#random forest accuracy = 0.965034965034965

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
print("Decision Tree accuracy:",dt.score(X_test,y_test))
#Decision Tree Accuracy : 0.9020979020979021

#to limit max feature (default : square root of) to 5
rf = RandomForestClassifier(max_features=5)

#to change estimator (decision tree) to 15
rf = RandomForestClassifier(n_estimators=15)

#parameter grid, to vary and give a list of the values to try
param_grid = {
    'n_estimator': [10,25,50,75,100]
}

rf = RandomForestClassifier()
gs = GridSearchCV(rf, param_grid, cv=5)

gs.fit(X,y)
print("best params:",gs.best_params_)
#Best params: {'n_estimator': 50}

### ELBOW GRAPH ###
"""
Elbow Graph is a model that optimizes performance without adding unnecessary complexity.

To find the optimal value, lets do a Grid Search trying all the values from 1 to 100 for n_estimators.
"""
n_estimators = list(range(1,101))
param_grid = {
    'n_estimators' : n_estimators,
}
rf = RandomForestClassifier()
gs = GridSearchCV(rf,param_grid,cv=5)
gs.fit(X,y)

scores = gs.cv_results_['mean_test_score']
# [0.91564148, 0.90685413, ...]

"""
Now lets use matplotlib to graph the results.
"""

import matplotlib.pyplot as plt

scores = gs.cv_results_['mean_test_score']
plt.plot(n_estimators, scores)
plt.xlabel("n_estimators")
plt.ylabel("accuracy")
plt.xlim(0, 100)
plt.ylim(0.9, 1)
plt.show()

# Now we can build our random forest model with the optimal number of trees.
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X, y) 

#feature importances
rf = RandomForestClassifier(n_estimators=10, random_state=111)
rf.fit(X_train, y_train)

ft_imp = pd.Series(rf.feature_importances_, index=cancer_data.feature_names).sort_values(ascending=False)
ft_imp.head(10)

"""
From the output, we can see that among all features, worst radius is most important (0.31), 
followed by mean concave points and worst concave points.
"""

"""
In our dataset, we happen to notice that features with "worst" seem to have higher importances. 
As a result we are going to build a new model with the selected features and see if it improves accuracy. 
Recall the model from the last part.
"""


rf = RandomForestClassifier(n_estimators=10, random_state=111)
rf.fit(X_train,y_train)
rf.score(X_test,y_test)
#0.965........

worst_cols = [col for col in df.columns if 'worst' in col]
print(worst_cols)

#There are ten such features. Now we create another dataframe with the selected features, followed by a train test split with the same random state.
X_worst = df[worst_cols]
X_train, X_test, y_train, y_test = train_test_split(X_worst, y, random_state=101)

#fit the model and output the accuracy
rf.fit(X_train,y_train)
rf.score(X_test,y_test)
