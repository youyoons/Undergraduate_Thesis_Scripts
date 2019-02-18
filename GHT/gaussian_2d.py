from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import math

'''
x_values = np.linspace(0, 58, 58)
y_values = np.linspace(0, 68, 68)

mu_x = 29
sig_x = 5.348265
g_x = np.exp(-np.power(x_values - mu_x, 2.) / (2 * np.power(sig_x, 2.)))
g_x = g_x*100
print(g_x)
print(np.shape(g_x))

mu_y = 34
sig_y = 8.75213
g_y = np.exp(-np.power(y_values - mu_y, 2.) / (2 * np.power(sig_y, 2.)))
g_y = g_y * 100
print(g_y)


g = np.outer(g_x,g_y)
print(np.shape(g))

plt.imshow(g)
plt.show()
'''
'''
mean = [29,34]
cov = [[5.5,0],[0,8.5]] 

x,y = np.random.multivariate_normal(mean,cov,5000).T
plt.plot(x,y,'x')
plt.axis('equal')
plt.show()

print(x[0:30])
'''
x_mu = 29
x_sig = 5.348

y_mu = 34
y_sig = 8.752

x, y = np.mgrid[0:58,0:68]
print(x[0:6,0:6])
x_pwr = (x - x_mu)**2/(2*x_sig**2)
y_pwr = (y - y_mu)**2/(2*y_sig**2)

prior = (1/(2*math.pi*x_sig*y_sig))*np.exp(-(x_pwr+y_pwr))
#prior = np.log(prior)
plt.gray()
plt.imshow(prior)
plt.show()

print(prior[23:30,24:34])
#print(prior[0:5,0:5])
print(np.shape(prior))
