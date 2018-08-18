import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.linear_model import LogisticRegression

# sigmoid function
# p(i) = 1 / (1 + exp[-(b0+b1*x)])

X = np.array([[10000, 80000, 35], [7000, 120000, 57], [100, 23000, 22], [223, 18000, 26]]) # balance, income, age
y = np.array([1, 1, 0, 0]) # if 1, they can pay bak the loan they want to take

classifier = LogisticRegression()
classifier.fit(X, y)

print( classifier.predict([[5500, 50000, 25]]))