import numpy as np
import matplotlib.pyplot as plt

# ФУНКЦИИ
def e(phi, h, t):
    return 0.5*np.sin(phi)**2 - h*np.cos(t - phi)

def de(phi, h, t):
    return 0.5*np.sin(2*phi) - h*np.sin(t - phi)

def d2e(phi, h, t):
    return np.cos(2*phi) + h*np.cos(t - phi)


# МЕТОД НЬЮТОНА
def newton(h, t, phi0, eps=1e-8, it=100):
    phi = phi0
    for _ in range(it):
        f  = de(phi, h, t)
        fp = d2e(phi, h, t)
        if fp <= 1e-10:
            return phi, False
        new = phi - f/fp
        if abs(new - phi) < eps:
            return new, True
        phi = new
    return phi, False


def find_min(h, t, phi):
    p, ok = newton(h, t, phi)
    if ok:
        return p, True
    best = None
    for trial in [phi + np.pi/2, phi - np.pi/2,
                  phi + np.pi,   -phi,
                  phi + 0.3,     phi - 0.3]:
        p, ok2 = newton(h, t, trial)
        if ok2:
            if best is None or e(p, h, t) < e(best, h, t):
                best = p
    if best is not None:
        return best, True
    return phi, False


def branch_one_way(t, hs, phi0):
    phi = phi0
    H, M = [], []
    for h in hs:
        phi, ok = find_min(h, t, phi)
        if ok:
            H.append(h)
            M.append(np.cos(t - phi))
    return np.array(H), np.array(M), phi


def full_loop(theta_deg, hmax=2.0, dh=0.001):
    t = np.deg2rad(theta_deg)
    hs_down = np.arange(hmax, -hmax - dh, -dh)
    hs_up   = np.arange(-hmax, hmax + dh, dh)

    H1, M1, phi_end = branch_one_way(t, hs_down, t)
    H2, M2, _       = branch_one_way(t, hs_up, phi_end)

    H = np.concatenate([H1, H2])
    M = np.concatenate([M1, M2])
    return H, M


def coercive_force(theta_deg):

    if theta_deg == 0:
        return 1.0
    if theta_deg >= 90:
        return 0.0
    t = np.deg2rad(theta_deg)
    return 1.0 / (np.sin(t)**(2/3) + np.cos(t)**(2/3))**1.5


# ПЕТЛИ

plt.figure(figsize=(8, 6))

for ang in [30, 60, 90]:
    h, m = full_loop(ang)
    plt.plot(h, m, lw=2.5, label=f'θ = {ang}°')

h0 = np.array([-2, -1, -1, 1, 1, 2])
m0 = np.array([-1, -1,  1, 1, -1, -1])
plt.plot(h0, m0, lw=2.5, label='θ = 0°')

plt.xlabel('h', fontsize=13)
plt.ylabel('m', fontsize=13)
plt.title('Петли гистерезиса m(h)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.axhline(0, color='k', lw=0.8)
plt.axvline(0, color='k', lw=0.8)
plt.show()


# АСТРОИДА

phi = np.linspace(0, 2*np.pi, 1000)
hx = -np.sin(phi)**3
hz =  np.cos(phi)**3

plt.figure(figsize=(6, 6))
plt.plot(hx, hz, 'b-', lw=3)
plt.xlabel('hx', fontsize=13)
plt.ylabel('hz', fontsize=13)
plt.title('Астроида Стопера-Вольфарта', fontsize=14)
plt.grid(True)
plt.axis('equal')
plt.axhline(0, color='k', lw=0.8)
plt.axvline(0, color='k', lw=0.8)
plt.show()


# ГРАФИК 3: КОЭРЦИТИВНАЯ СИЛА
angles = np.arange(0, 91, 2)
hc_loop = [coercive_force(a) for a in angles]

plt.figure(figsize=(8, 5))
plt.plot(angles, hc_loop, 'bo-', lw=2.5, ms=4)
plt.xlabel('θ, градусы', fontsize=13)
plt.ylabel('hc (коэрцитивная сила)', fontsize=13)
plt.title('Зависимость коэрцитивной силы от угла', fontsize=14)
plt.grid(True)
plt.show()




print("\nКоэрцитивная сила hc по формуле (13):")
print("-" * 40)
for a in [0, 30, 45, 60, 90]:
    print(f"θ = {a}°: hc = {coercive_force(a):.3f}")