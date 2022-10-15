import matplotlib.pylot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X,y = load_digits(n_class=2, return_X_y=True)

#drawing the digit
plt.matshow(X[0].reshape(8,8), cmap=plt.cm.gray)
plt.xtics(())
plt.ytics(())
plt.show()


#print the pixel image
print(X.shape, y.shape)
print(X[0])
print(y[0])
print(X[0].reshape(8,8))

#MLP for MNIST Dataset and draw it
X_train, x_test, y_train, y_test = train_test_split(X,y,random_state=2)
mlp = MLPClassifier
mlp.fit(X_train, y_train)

x = X_test[0]
plt.matshow(X[0].reshape(8,8), cmap=plt.cm.gray)
plt.xtics(())
plt.ytics(())
plt.show()

print(mlp.predict([x]))
