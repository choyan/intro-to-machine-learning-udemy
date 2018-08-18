import  numpy as np 
from matplotlib import pyplot as plt 
from sklearn.linear_model import LogisticRegression

# sigmoid function
# p(i) = 1 / (1 + exp[-(b0+b1*x)])


# data
x1 = np.array([0, 0.6, 1.1, 1.6, 1.0, 2.5, 3, 3.1, 3.9, 4, 4.9, 5, 5.1])
y1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

x2 = np.array([3, 3.8, 4.4, 5.2, 5.5, 6.5, 6, 6.1, 6.9, 7, 7.9, 8, 8.1])
y2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

x = ([ [0], [0.6], [1.1], [1.5], [1.8], [2.5], [3], [3.1], [3.9], [4], [4.9], [5], [5.1], [3], [3.8], [4.4], [5.2], [5.5], [6.5], [6], [6.5], [6.9], [7], [7.9], [8], [8.1] ])
y = ([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ])

plt.plot(x1, y1, 'ro', color='blue')
plt.plot(x2, y2, 'ro', color='red')


classifier = LogisticRegression()
classifier.fit(x, y)

# pred = classifier.predict(8)
pred = classifier.predict_proba(8) # Prints the probability of 87% 
print(pred)


# application of sigmoid function to draw a curved line
def model (classifier, x):
    return 1 / (1 + np.exp(-(classifier.intercept_ + classifier.coef_ * x))) # interept as b0 and coef as b1

for i in range(1, 120, 1):
    plt.plot(i/10.0 - 2, model(classifier, i/10.0-2), 'ro', color='green')


plt.axis([-2, 10, -0.5, 2])
plt.show()