import matplotlib as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

X,y = fetch_openml('mnist_784', version=1, return_X_y=True)

# print the shape of array and the reange ot the features value
print(X.shape, y.shape)
print(np.min(X), np.max(X))
print(y[0:5])

# only using digit 0-3
X5 = X[y <= '3']
y5 = y[y <= '3']

mlp = MLPClassifier(hidden_layer_sizes=(6,),max_iter=200, alpha=1e-4, solver='sgd', random_state=2)
mlp.fit(X5, y5)

# MLPClassifier Coefficients
print(mlp.coefs_)
print(len(mlp.coefs_))
print(mlp.coefs_[0].shape)

# Create multiple plots within a single plot
fig, axes = plt.subplots(2,3, figsize=(5,4))
for i, ax in enumerate(axes.ravel()):
  coef = mlp.coefs_[0][:,i]
  ax.matshow(coef.reshape(28,28), cmap=plt.cm.gray)
  ax.set_xstick(())
  ax.set_ystick(())
  ax.set_title(i+1)
 plt.show()
