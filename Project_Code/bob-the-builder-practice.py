#Thanks to Sanjai R in this forum page 
# url source = https://www.sololearn.com/Discuss/2634017/machine-learning-bob-the-builder-task-how-to-make

import numpy as np
from sklearn.linear_model import LogisticRegression
n = int(input())
X = []
for i in range(n):
    X.append([float(x) for x in input().split()])
y = [int(x) for x in input().split()]
datapoint = [float(x) for x in input().split()]
model = LogisticRegression()
model.fit(X,y)
datapoint = np.array(datapoint).reshape(1,-1)
predicting = model.predict(datapoint[[0]])[0]
print(predicting)