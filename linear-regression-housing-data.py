from scipy import stats
import numpy as np
from matplotlib import pyplot as plt 

x = np.array([112, 345, 190, 305, 372, 550, 302, 440, 578])
y = np.array([1120, 1523, 2102, 2230, 2600, 3200, 3409, 3689, 4460])

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plt.plot(x, y, 'ro', color='black')
plt.ylabel('Price')
plt.xlabel('Size of the house')

plt.axis([0, 500, 0, 5000])

plt.plot(x, x*slope+intercept, 'b')
plt.plot()

plt.show()


# Prediction
dataX = 130
prediction = (dataX*slope) + intercept
print(prediction)