import pandas as pd
from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()

#showing object in dataset
print(cancer_data.keys())
print(cancer_data['DESCR'])

#Load data into pandas dataframe
df = pd.DataFrame(cancer_data['data'])
columns=cancer_data['feature_names']
print(df.head())

#build logistic regression model
X = df[cancer_data.feature_name].values
y = df['target'].values

model = LogisticRegression()
model.fit(X,y)
prediction_model = model.predict([X[0]])
score_model = model.score(X,y)
print(prediction_model)
print(score_model)
