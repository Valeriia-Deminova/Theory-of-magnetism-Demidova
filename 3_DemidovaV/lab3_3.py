# задание 3
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
xd = np.linspace(-L, L, 4000)

H_over_HK = [0, 0.02, 0.05, 0.1, 0.15, 0.2]
ratios = []
print("H/HK     delta/delta0")
for hr in H_over_HK:
    th = solve_bvp(make_ode(hr*HK), bc, x, y0, tol=1e-8).sol(xd)[0]
    i1 = np.argmin(np.abs(th - 0.1*np.pi))
    i2 = np.argmin(np.abs(th - 0.9*np.pi))
    d = (xd[i2] - xd[i1])/delta0
    ratios.append(d)
    print("%5.2f     %.4f" % (hr, d))


C = (ratios[-1] - 1.0) / (H_over_HK[-1]**2)
hh = np.linspace(0, 0.2, 100)
theory = 1.0 + C * hh**2

plt.plot(H_over_HK, ratios, 'bo-', label='Численное решение')
plt.plot(hh, theory, 'r--', label=r'Оценка $O(H_\perp^2/H_K^2)$')
plt.xlabel(r'$H_\perp/H_K$')
plt.ylabel(r'$\delta/\delta_0$')
plt.title('Ширина границы в поперечном поле')
plt.legend()
plt.grid()

#plt.savefig('задание 3.png', dpi=150)
plt.show()