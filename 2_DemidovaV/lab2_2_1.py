import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, np.pi, 120)
phi   = np.linspace(0, 2*np.pi, 240)
T, P  = np.meshgrid(theta, phi)

ax_ = np.sin(T)*np.cos(P)
ay_ = np.sin(T)*np.sin(P)
az_ = np.cos(T)

def draw(w, name):
    r = w - w.min() + 0.05
    X = r*np.sin(T)*np.cos(P)
    Y = r*np.sin(T)*np.sin(P)
    Z = r*np.cos(T)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(w/w.max()),
                    rstride=2, cstride=2, linewidth=0, antialiased=True)
    ax.set_title(name)
    plt.show()

draw(np.sin(T)**2, 'a) K1>0')
draw(-np.sin(T)**2, 'b) K1<0')
draw(ax_**2*ay_**2 + ay_**2*az_**2 + az_**2*ax_**2, 'c) K1c>0')
draw(-(ax_**2*ay_**2 + ay_**2*az_**2 + az_**2*ax_**2), 'd) K1c<0')