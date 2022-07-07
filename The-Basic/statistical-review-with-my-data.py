import numpy as np

data = []
file = open("statistical-number.txt","r")
for number in file.readlines():
    for i in number.split():
        data.append(int(i))

print("mean : ",np.mean(data))
print("median : ",np.median(data))
print("50th percentile (median) : ", np.percentile(data,50))
print("75th percentile : ", np.percentile(data,75))
print("25th percentile : ", np.percentile(data,25))
print("standard deviation : ",np.std(data))
print("variance : ",np.var(data))