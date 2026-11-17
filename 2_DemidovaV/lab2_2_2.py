import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(1e-3, np.pi-1e-3, 140)
phi   = np.linspace(0, 2*np.pi, 280)
T, P  = np.meshgrid(theta, phi)

ax_ = np.sin(T)*np.cos(P)
ay_ = np.sin(T)*np.sin(P)
az_ = np.cos(T)
w_cub = ax_**2*ay_**2 + ay_**2*az_**2 + az_**2*ax_**2

def draw(w, name):
    r = w - w.min() + 0.05
    X = r*np.sin(T)*np.cos(P)
    Y = r*np.sin(T)*np.sin(P)
    Z = r*np.cos(T)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.plasma(w/w.max()),
                    rstride=3, cstride=3, linewidth=0, antialiased=True)
    ax.set_title(name)
    plt.show()

# 2.2: ось формы вдоль z
for ke in [0.1, 0.5, 1.0, 2.0]:
    draw(w_cub + ke*np.sin(T)**2,
         rf'$\mathbf{{n}}\parallel z$,  $K_{{ef}}/K_{{1c}} = {ke}$')

# 2.3: ось формы (pi/4, pi/4)
nx, ny, nz = 0.5, 0.5, 1/np.sqrt(2)
proj2 = (ax_*nx + ay_*ny + az_*nz)**2
for ke in [0.1, 0.5, 1.0, 2.0]:
    draw(w_cub + ke*(1 - proj2),
         rf'$\theta_n=\pi/4,\ \varphi_n=\pi/4$,  $K_{{ef}}/K_{{1c}} = {ke}$')