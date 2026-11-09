# задание 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

A, K, Ms = 2.0e-11, 4.0e4, 1.7e6
mu0 = 4*np.pi*1e-7
Delta0 = np.sqrt(A/K); delta0 = np.pi*Delta0
HK = 2*K/(mu0*Ms)

def make_ode(Hp):
    def ode(x, y):
        theta, dtheta = y
        return np.vstack((dtheta,
            (K/A)*np.sin(theta)*np.cos(theta) - (mu0*Ms*Hp/(2*A))*np.cos(theta)))
    return ode

def bc(ya, yb):
    return np.array([ya[0], yb[0] - np.pi])

L = 5*delta0
x = np.linspace(-L, L, 400)
th0 = 2*np.arctan(np.exp(x/Delta0))
y0 = np.vstack((th0, np.gradient(th0, x)))
xd = np.linspace(-L, L, 2000)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for hr, lab in [(0, '0'), (0.05, '0.05HK'), (0.1, '0.1HK'), (0.2, '0.2HK')]:
    sol = solve_bvp(make_ode(hr*HK), bc, x, y0, tol=1e-8)
    th = sol.sol(xd)[0]
    xc = xd[np.argmin(np.abs(th - np.pi/2))]
    print("H/HK=%-6s  x_center = %+.3f нм" % (lab, xc*1e9))
    ax[0].plot(xd/1e-9, th/np.pi, label=lab)
    ax[1].plot(xd/1e-9, np.cos(th), label=lab)

ax[0].set_xlabel('x, нм'); ax[0].set_ylabel(r'$\theta/\pi$'); ax[0].legend(); ax[0].grid()
ax[1].set_xlabel('x, нм'); ax[1].set_ylabel(r'$M_z/M_s$'); ax[1].legend(); ax[1].grid()
plt.tight_layout()

#plt.savefig('задание 2.png', dpi=150)
plt.show()