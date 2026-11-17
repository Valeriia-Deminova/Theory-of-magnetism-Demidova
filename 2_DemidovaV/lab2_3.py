import numpy as np
import matplotlib.pyplot as plt


theta = np.linspace(0.08, np.pi - 0.08, 180)
phi   = np.linspace(0, 2*np.pi, 360, endpoint=False)
T, P  = np.meshgrid(theta, phi)

ax_ = np.sin(T)*np.cos(P)
ay_ = np.sin(T)*np.sin(P)
az_ = np.cos(T)

w_uni = np.sin(T)**2
w_cub = ax_**2*ay_**2 + ay_**2*az_**2 + az_**2*ax_**2


def draw(w, name):
    r = w - w.min() + 0.05
    X = r*np.sin(T)*np.cos(P)
    Y = r*np.sin(T)*np.sin(P)
    Z = r*np.cos(T)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.coolwarm(w/w.max()),
                    rstride=4, cstride=4, linewidth=0, antialiased=True)
    ax.set_title(name)
    plt.show()


for h in [0, 0.3, 0.7, 4.0]:
    draw(w_uni - 2*h*ax_, rf'3.1  $h = {h}$')


for h in [0, 0.3, 0.7, 4.0]:
    draw(w_cub  - 2*h*ax_, rf'3.2  $K_{{1c}} > 0$,  $h = {h}$')
    draw(-w_cub - 2*h*ax_, rf'3.2  $K_{{1c}} < 0$,  $h = {h}$')