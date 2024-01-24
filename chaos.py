import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


def lorenz(p, t, s, r, b):
    x, y, z = p.tolist()
    return s * (y - x), x * (r - z) - y, x * y - b * z  # 返回dx/dt,dy/dt,dz/dt
    # return y, -x * z, x * y  # 返回dx/dt,dy/dt,dz/dt


para1 = (10.0, 13.926, 2.67)  # sigma, r, b
para2 = (10.0, 28.0, 2.67)
# para1 = (5.0, 23.5, 2.0)  # 14
# para2 = (5.0, 23.0, 2.0)
init1 = (0.0, 1.0, 0.0)
# init1 = (6.83, 6.83, 23.9)
init2 = (0.0, 1.0 - 1e-10, 0.0)
t = np.arange(0, 100, 0.005)
track1 = integrate.odeint(lorenz, init1, t, args=para1)
track2 = integrate.odeint(lorenz, init2, t, args=para2)


def ellipsoid_surface(phi, theta, s, r, C):
    x = np.sqrt(C / r) * np.sin(theta) * np.cos(phi)
    y = np.sqrt(C / s) * np.sin(theta) * np.sin(phi)
    z = 2 * r + np.sqrt(C / s) * np.cos(theta)
    return x, y, z


def plot_Lorenz():
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('Lorenz fig')
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(track1[:, 0], track1[:, 1], track1[:, 2], lw=1.0, color='r')  # 画轨迹1
    ax1.set_title(f"$(\\sigma, r, b)={para1}$\n$(x_0,y_0,z_0)={init1}$")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    phi = np.linspace(0, 2 * np.pi, 100)
    theta = np.linspace(0, np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)
    x, y, z = ellipsoid_surface(phi, theta, para1[0], para1[1],
                                (para1[1] ** 2) * max([4 * para1[0], para1[0] * (para1[2] + 1), para1[0] + para1[2]]))
    # ax1.plot_surface(x, y, z, color='b', alpha=0.3)
    # ax2 = fig.add_subplot(122)
    # ax2.plot(t, track1[:, 1], lw=1.0, color='b')
    # ax2.set_xlabel('t')
    # ax2.set_ylabel('y')
    # ax2.set_title('relation between $t$ and $y$')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(track2[:, 0], track2[:, 1], track2[:, 2], lw=1.0, color='g')  # 画轨迹2
    ax2.set_title(f"$(\\sigma, r, b)={para2}$\n$(x_0,y_0,z_0)={init2}$")
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    plt.show()


def plot_t_y():
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('relation between $t$ and $y$')
    ax1 = fig.add_subplot(121)
    ax1.plot(t, track1[:, 1], lw=1.0, color='r')  # 画轨迹1
    ax1.plot(t, [0] * len(t), '--', lw=1.0, color='gray')
    ax1.set_title(f"$(\\sigma, r, b)={para1}$\n$(x_0,y_0,z_0)={init1}$")
    ax1.set_xlabel('t')
    ax1.set_ylabel('y')
    ax2 = fig.add_subplot(122)
    ax2.plot(t, track2[:, 1], lw=1.0, color='g')  # 画轨迹2
    ax2.plot(t, [0] * len(t), '--', lw=1.0, color='gray')
    ax2.set_title(f"$(\\sigma, r, b)={para2}$\n$(x_0,y_0,z_0)={init2}$")
    ax2.set_xlabel('t')
    ax2.set_ylabel('y')
    plt.show()

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('relation between $t$ and $y$')
    ax1 = fig.add_subplot(111)
    ax1.plot(t, track1[:, 1], lw=1.0, color='r')  # 画轨迹1
    ax1.plot(t, track2[:, 1], lw=1.0, color='g')  # 画轨迹2
    ax1.plot(t, [0] * len(t), '--', lw=1.0, color='gray')
    ax1.set_xlabel('t')
    ax1.set_ylabel('y')
    plt.show()


def plot_Zn():
    Z = track1[:, 2]
    t_zn = (Z[1:-1] > Z[:-2]) & (Z[1:-1] > Z[2:])
    t_zn = np.where(t_zn)[0] + 1

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('relation between $t$ and $z$')
    ax1 = fig.add_subplot(111)
    ax1.plot(t, track1[:, 2], lw=1.0, color='r')  # 画轨迹1
    ax1.scatter(t[t_zn], track1[:, 2][t_zn], s=15)
    ax1.plot(t, [0] * len(t), '--', lw=1.0, color='gray')
    ax1.set_xlabel('t')
    ax1.set_ylabel('z')
    plt.show()

    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    ax1.scatter(Z[t_zn[:-1]], Z[t_zn[1:]], s=10)
    ax1.plot([25, 50], [25, 50], '--', lw=1.0, color='gray')
    ax1.set_xlabel('$Z_n$')
    ax1.set_ylabel('$Z_{n+1}$')
    plt.show()


def plot_delta():
    # track1 = np.array(track1)
    # track2 = np.array(track2)
    delta = np.sqrt(
        (track1[:, 0] - track2[:, 0]) ** 2 + (track1[:, 1] - track2[:, 1]) ** 2 + (track1[:, 2] - track2[:, 2]) ** 2
    )
    delta = np.log(delta)
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('relation between $t$ and $ln(|\delta|)$')
    ax1 = fig.add_subplot(111)
    ax1.plot(t, delta, '-', lw=1.0, color='b')
    ax1.set_xlabel('t')
    ax1.set_ylabel('$ln(|\delta|)$')
    ax1.plot([0, 30], [-23, 2], '--', lw=1.5, color='black')
    plt.show()


if __name__ == '__main__':
    plot_Lorenz()
    # plot_t_y()
    # plot_Zn()
    # plot_delta()
