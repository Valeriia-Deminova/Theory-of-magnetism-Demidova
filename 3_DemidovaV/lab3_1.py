# задание 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

A = 2.0e-11
K = 4.0e4
Ms = 1.7e6
mu0 = 4*np.pi*1e-7
Delta0 = np.sqrt(A/K)
delta0 = np.pi*Delta0

def ode(x, y):
    theta, dtheta = y
    return np.vstack((dtheta, (K/A)*np.sin(theta)*np.cos(theta)))

def bc(ya, yb):
    return np.array([ya[0], yb[0] - np.pi])

L = 5*delta0
x = np.linspace(-L, L, 400)
th0 = 2*np.arctan(np.exp(x/Delta0))
sol = solve_bvp(ode, bc, x, np.vstack((th0, np.gradient(th0, x))), tol=1e-8)

xd = np.linspace(-L, L, 2000)
theta = sol.sol(xd)[0]
theta_an = 2*np.arctan(np.exp(xd/Delta0))

print("delta0(формула) = %.2f нм" % (delta0*1e9))
i1 = np.argmin(np.abs(theta - 0.1*np.pi))
i2 = np.argmin(np.abs(theta - 0.9*np.pi))
print("delta(числ.)    = %.2f нм" % ((xd[i2]-xd[i1])*1e9))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(xd/1e-9, theta/np.pi, 'b-', lw=2, label='числ.')
ax[0].plot(xd/1e-9, theta_an/np.pi, 'r--', lw=2, label='аналит.')
ax[0].set_xlabel('x, нм'); ax[0].set_ylabel(r'$\theta/\pi$'); ax[0].legend(); ax[0].grid()

ax[1].plot(xd/1e-9, np.cos(theta), 'b-', lw=2, label=r'$M_z/M_s$')
ax[1].plot(xd/1e-9, np.sin(theta), 'g-', lw=2, label=r'$M_x/M_s$')
ax[1].set_xlabel('x, нм'); ax[1].set_ylabel('M/Ms'); ax[1].legend(); ax[1].grid()

plt.tight_layout()
#plt.savefig('задание 1.png', dpi=150)
plt.show()

