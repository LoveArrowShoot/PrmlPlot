import numpy as np
import matplotlib.pyplot as plt

# Redefine map under python3
amap = lambda func, *iterable: np.array(list(map(func, *iterable)))

training_data_num = 10
training_data_x = np.linspace(0,1,training_data_num)
training_data_t = np.add(amap(np.sin, training_data_x*2*np.pi), np.random.normal(0,0.3,training_data_num))

# Minimize error, exercise 1.1
def fitting(x, t, M):
    A = np.zeros((M + 1,M + 1))
    T = np.zeros(M + 1)
    for i in range(M + 1):
        T[i] = sum(np.multiply(t, amap(lambda x: np.power(x,i), x)))
        for j in range(M + 1):
            A[i,j] = sum(amap(lambda x: np.power(x,i + j), x))
    return np.linalg.solve(A,T)

# Generate polynomial y(x,w)
def polynomial(w,M):
    return lambda x: sum(w * amap(lambda i:np.power(x,i), range(M + 1)))

# Draw graph
polynomial_order = 5
w = fitting(training_data_x, training_data_t, polynomial_order)
y = polynomial(w, polynomial_order)

plot_x = np.linspace(-0.05, 1.05, 100)
plot_y = amap(y, plot_x)

plt.clf()
plt.axis([-0.05, 1.05, -1.49, 1.49])
plt.scatter(training_data_x, training_data_t)
plt.plot(plot_x, amap(np.sin, plot_x*2*np.pi), 'g-')
plt.plot(plot_x, plot_y, 'r-')
plt.show()