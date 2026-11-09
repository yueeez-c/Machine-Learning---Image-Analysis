import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    return 2 - np.exp(-4*t) - 2*y

def exact(t):
    return 1 - 0.5*np.exp(-2*t) + 0.5*np.exp(-4*t)

def euler(h):
    t = np.arange(0, 5 + h, h)
    y = np.zeros(len(t))
    y[0] = 1

    for n in range(len(t) - 1):
        y[n+1] = y[n] + h*f(t[n], y[n])

    return t, y


# Exact solution
t_exact = np.linspace(0, 5, 1000)
plt.plot(t_exact, exact(t_exact),
         'k--', linewidth=2, label='Exact')

# Euler approximations
for h in [1]:
    t, y = euler(h)
    plt.plot(t, y, label=f'h = {h}')

plt.xlabel('t')
plt.ylabel('y(t)')
plt.title("Euler's Method with h=1")
plt.legend()
plt.grid()
plt.show()