import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import Image
#Cross Validation
from sklearn.model_selection import GridSearchCV


#dataset before
df = pd.read_csv("https://sololearn.com/uploads/files/titanic.csv")
df['male'] = df['Sex'] == 'male'


feature_names = ['Pclass', 'male']
X = df[feature_names].values
y = df['Survived'].values

#Classic Decision Tree
#dt = DecisionTreeClassifier()

#set prepruning techniques
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, max_leaf_nodes=10)
dt.fit(X, y)

#param grid variable
param_grid = {
    'max_depth': [5, 15, 25],
    'min_samples_leaf': [1, 3],
    'max_leaf_nodes': [10, 20, 35, 50]}
dt = DecisionTreeClassifier()
gs = GridSearchCV(dt, param_grid, scoring='f1', cv=5)
gs.fit(X, y)

print("best paramas:",gs.best_params_)
print("best score:", gs.best_score_)

dot_file = export_graphviz(dt, feature_names=feature_names)
graph = graphviz.Source(dot_file)
graph.render(filename='tree', format='png', cleanup=True)