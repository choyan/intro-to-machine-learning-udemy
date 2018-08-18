import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier


# data
xFruit = np.array([10, 10])
yFruit = np.array([9, 1])

xProtein = np.array([1, 1])
yProtein = np.array([4, 1])

xVeg = np.array([7])
yVeg = np.array([10])


X = np.array([[10, 9], [10, 1], [1, 4], [1, 1], [7, 10]])
y = np.array([0, 0, 1, 1, 2]) # 0: Fruit, 1: Protein, 2: Vegs

plt.plot(xFruit, yFruit, 'ro', color='blue')
plt.plot(xProtein, yProtein, 'ro', color='red')
plt.plot(xVeg, yVeg, 'ro', color='yellow')

plt.plot(6, 4, 'ro', color='green', markersize=15)
plt.axis([-0.5, 15, -0.5, 15])

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X, y)

pred = classifier.predict([[6, 4]])
#pred = classifier.predict_proba(8) # Prints the probability of 87% 
print(pred)

plt.show()