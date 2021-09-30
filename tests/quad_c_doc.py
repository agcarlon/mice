from mice import MICE, plot_mice
import numpy as np
import matplotlib.pyplot as plt

kappa = 100
H_aux = np.array([[kappa, kappa-1], [kappa-1, kappa]])
b = np.ones(2)


def dobjf(x, thetas):
    gradients = []
    for theta in thetas:
        H = np.eye(2) * (1 - theta) + H_aux * theta
        gradients.append(H @ x.T - b)
    return np.vstack(gradients)


def sampler(n):
    return np.random.uniform(0, 1, int(n))


df = MICE(dobjf,
          sampler=sampler,
          eps=.7,
          max_cost=1e4,
          min_batch=5)

x = np.array([20., 50.])
L = kappa
mu = 1
step_size = 2 / (L + mu) / (1 + df.eps ** 2)

EH = H_aux*.5 + .5*np.eye(2)
opt = np.linalg.solve(EH, b)

while True:
    grad = df.evaluate(x)
    if df.terminate:
        break
    x = x - step_size * grad

fig, ax = plt.subplots(figsize=(6, 5))
ax = plot_mice(df.log, ax, 'iteration', 'grad_norm', style='semilogy')
ax.set_xlabel('Iteration')
ax.set_ylabel('Norm of estimate')
fig.savefig('usage_ex.png')
print('End')