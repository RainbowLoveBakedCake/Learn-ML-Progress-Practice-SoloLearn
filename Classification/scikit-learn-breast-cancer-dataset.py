import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
cancer_data = load_breast_cancer()

#showing object in dataset
print(cancer_data.keys())
print(cancer_data['DESCR'])

#Load data into pandas dataframe
df = pd.DataFrame(cancer_data['data'])
columns=cancer_data['feature_names']
print(df.head())

#build logistic regression model
X = df[cancer_data.feature_names].values
y = df['target'].values

model = LogisticRegression()
model.fit(X,y)
prediction_model = model.predict([X[0]])
score_model = model.score(X,y)
print(prediction_model)
print(score_model)
