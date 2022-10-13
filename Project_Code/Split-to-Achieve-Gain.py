# Source = https://www.youtube.com/watch?v=CIkdNH1qCtI

S = [int(x) for x in input().split()]
A = [int(x) for x in input().split()]
B = [int(x) for x in input().split()]

S_length = len(S)
A_length = len(A)
B_length = len(B)

Number_one = S.count(1)
Number_zero = S.count(0)
Gini = Number_one/(Number_one+Number_zero)
Gini_impurity_S = 2*Gini*(1-Gini)

Number_one = A.count(1)
Number_zero = A.count(0)
Gini = Number_one/(Number_one+Number_zero)
Gini_impurity_A = 2*Gini*(1-Gini)

Number_one = B.count(1)
Number_zero = B.count(0)
Gini = Number_one/(Number_one+Number_zero)
Gini_impurity_B = 2*Gini*(1-Gini)

Information_Gain = Gini_impurity_S-(Gini_impurity_A*(A_length/S_length))-(Gini_impurity_B*(B_length/S_length))
print(round(Information_Gain,5))