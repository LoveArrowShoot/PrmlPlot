import matplotlib.pyplot as plt
import numpy as np
import math
from functools import reduce

training_data_num = 10
training_data_x = np.linspace(0, 2*math.pi, training_data_num)
training_data_t = np.add(np.sin(training_data_x), np.random.uniform(-0.2, 0.2, training_data_num))

# Gaussian kernel
sigma = 0.5
def kernel(x1, x2):
   return np.exp(-np.power(x1-x2,2)/(2*np.power(sigma, 2)))

# Exponential kernel
# theta = 2
# def kernel(x1, x2):
#     return np.exp(-theta*np.abs(x1-x2))

def calcute(a, b, f):
    rows = len(a)
    cols = len(b)
    result = np.zeros([rows, cols])
    for r in range(rows):
        for c in range(cols):
            result[r,c] = f(a[:][r], b[:][c])
    return result

# Gram matrix
K = calcute(training_data_x, training_data_x, kernel)

# Covariance of marginal
beta = 25
C_N = K + np.eye(training_data_num)/beta

x = np.linspace(0, 2*math.pi, 1000)

# Mean of posterior
m = reduce(np.dot, [calcute(x, training_data_x, kernel), np.linalg.inv(C_N), np.transpose(training_data_t)])

# Draw graph
plt.clf()
plt.axis([0, 6.3, -1.3, 1.3])
plt.hold(True)
plt.plot(x, m)
plt.plot(x, np.sin(x))
plt.plot(training_data_x, training_data_t, 'o')
plt.hold(False)
plt.grid()
plt.show()