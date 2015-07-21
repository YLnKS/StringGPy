from samplers import SGPRegressor, SGPBinaryClassifier
import numpy as np
import pylab as plt

l = 1000
lb = 100.0
n = int(lb*l)
K = 100

data = np.zeros((n, 2))
data[:, 0]=np.arange(0.0, lb, 1.0/l)
data[:, 1]=np.sin(data[:, 0]*(2.0*np.pi)) + 0.10*np.random.normal(size=n)

m = SGPRegressor(data, [['se']*n], [[np.array([4.0, 0.5]).copy()]*n], [list(np.arange(0.0, lb+1.0/l, 1.0/l))], 0.01, "prod", should_print=True, sgp_bridge=True)
m.sample(n=800)
print np.mean(np.abs(m.f_samples[-1]-data[:, 1])), np.std(np.abs(m.f_samples[-1]-data[:, 1])), m.noise_var

plt.figure()
plt.plot(m.data[:, 0], m.f_samples[-1], 'b*', m.data[:, 0], m.data[:, 1], 'r*')
plt.show()

plt.figure()
plt.plot(m.data[:, 0], m.f_samples[-100], 'b*', m.data[:, 0], m.data[:, 1], 'r*')
plt.show()

plt.figure()
plt.plot(m.data[:, 0], m.f_samples[-200], 'b*', m.data[:, 0], m.data[:, 1], 'r*')
plt.show()


# l = 1000
# lb = 1.0
# n = int(lb*l)
# K = 100

# data = np.zeros((n, 2))
# data[:n/2, 0]= -1.0 + 0.6*np.random.normal(size=n/2)
# data[:n/2, 1]= 0.0
# data[n/2:, 0]= 1.0 + 0.6*np.random.normal(size=n/2)
# data[n/2:, 1]= 1.0


# m = SGPBinaryClassifier(data, [['se']*n], [[np.array([4.0, 2.0]).copy()]*n], [list(np.sort(data[:, 0])) + [max(data[:, 0]) + 0.01]], "prod", should_print=True, sgp_bridge=True)
# m.sample(n=2000)

# plt.figure()
# plt.plot(m.data[:, 0], 1.0/(1.0+np.exp(-m.f_samples[-1])), 'k*', data[:n/2, 0], data[:n/2, 1], 'r*', data[n/2:, 0], data[n/2:, 1], 'b*')
# plt.show()

# plt.figure()
# plt.plot(m.data[:, 0], 1.0/(1.0+np.exp(-m.f_samples[-100])), 'k*', data[:n/2, 0], data[:n/2, 1], 'r*', data[n/2:, 0], data[n/2:, 1], 'b*')
# plt.show()

# plt.figure()
# plt.plot(m.data[:, 0], 1.0/(1.0+np.exp(-m.f_samples[-200])), 'k*', data[:n/2, 0], data[:n/2, 1], 'r*', data[n/2:, 0], data[n/2:, 1], 'b*')
# plt.show()


# from time import time

# t = time()
# lst = [_ for _ in xrange(10000000)]
# dt = time() - t

# t = time()
# dct = {str(_):_ for _ in xrange(10000000)}
# dt2 = time() - t

# t = time()
# arr = np.zeros((10000000))
# dt3 = time() - t

# print 'Dict took', str(dt2), 'List took', str(dt), 'ndarray', str(dt3)