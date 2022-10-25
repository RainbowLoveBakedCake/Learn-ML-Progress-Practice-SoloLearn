import math

w1, w2, b, x1, x2 = [float(x) for x in input().split()]
r = (w1*x1) + (w2*x2) + b
z = math.exp(-r)
sigmoid = 1/(1+z)
sigmoid = round(sigmoid,4)
print(sigmoid)

##thanks to this discussion https://www.sololearn.com/Discuss/2755527/machine-learning-the-sigmoid-function/
