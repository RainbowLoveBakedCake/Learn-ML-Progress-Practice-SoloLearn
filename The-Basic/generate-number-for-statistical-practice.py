import random
from tokenize import Number

from numpy import number

file1 = open("statistical-number.txt","w")

for i in range(int(input('How many random number needed?: '))):
    RandomNumber = str(random.randint(1,1000))
    file1.write(RandomNumber)
    file1.write("\n")
    print(file1)
file1.close()
