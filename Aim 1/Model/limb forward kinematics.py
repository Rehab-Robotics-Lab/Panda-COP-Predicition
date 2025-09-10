import numpy as np
import math
from numpy import radians as radians
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transform_manager import TransformManager
from scipy.spatial.transform import Rotation as R

# time step seconds
dt = 10

# array of angles from robot
angle = radians(np.array([0, 10, 20, 30, 40, 50, 60]))

# array of all angles
n = len(angle)
theta = np.zeros((4, n))

# theta 2 is angles from robot and theta 4 is fixed elbow angle of about 30%
theta[1, :] = angle
theta[3, :] = np.ones_like(angle) * radians(30)

# angular velocity and acceleration
theta_d = np.gradient(theta, dt, axis=1)
theta_dd = np.gradient(theta_d, dt, axis=1)
# print(theta_dd)


def Ti(ai, alpha0, di, thetai):
    T = np.matrix(
        [
            [math.cos(thetai), -math.sin(thetai), 0, ai],
            [
                math.sin(thetai) * math.cos(alpha0),
                math.cos(thetai) * math.cos(alpha0),
                -math.sin(alpha0),
                -math.sin(alpha0) * di,
            ],
            [
                math.sin(thetai) * math.sin(alpha0),
                math.cos(thetai) * math.sin(alpha0),
                math.cos(alpha0),
                math.cos(alpha0) * di,
            ],
            [0, 0, 0, 1],
        ]
    )
    return T


def show_Ti(ax, Ti, color, i):
    Ri = Ti[0:3, 0:3]
    # print("R: ", Ri)
    print("T", str(i), ": ", Ti)

    origin = np.matrix([[0], [0], [0], [1]])

    vec = Ti @ origin
    vec = Ri.T @ vec[0:3, :]
    # print("dx: ", vec.T)

    dx, dy, dz = vec[0, 0], vec[1, 0], vec[2, 0]

    xaxis = np.matrix([[1], [0], [0]])
    yaxis = np.matrix([[0], [1], [0]])
    zaxis = np.matrix([[0], [0], [1]])

    x = Ri @ xaxis
    y = Ri @ yaxis
    z = Ri @ zaxis

    # print(x)

    # print(xaxis)

    # print(x)

    # [0,3], Ti[1,3], Ti[]
    ax.quiver([dx], [dy], [dz], x[0, 0], x[1, 0], x[2, 0], colors=color)
    ax.text(x[0, 0] + dx + 0.1, x[1, 0] + dy + 0.1, x[2, 0] + dz + 0.1, "X" + str(i), size=10, zorder=1, color=color)

    ax.quiver([dx], [dy], [dz], y[0, 0], y[1, 0], y[2, 0], colors=color)
    ax.text(y[0, 0] + dx + 0.2, y[1, 0] + dy + 0.2, y[2, 0] + dz + 0.2, "Y" + str(i), size=10, zorder=1, color=color)

    ax.quiver([dx], [dy], [dz], z[0, 0], z[1, 0], z[2, 0], colors=color)
    ax.text(z[0, 0] + dx + 0.3, z[1, 0] + dy + 0.3, z[2, 0] + dz + 0.3, "Z" + str(i), size=10, zorder=1, color=color)

    return ax


# make global/tensor (R,m,p)


# generic = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])\

# T_d1 = np.matrix(
#     [
#         [1, 0, 0, 0],
#         [0, 1, 0, -1.5],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1],
#     ]
# )


# T_b0 = Ti(0, 0, l0, 0)

# sholder abduction
thet1 = radians(0)
# sholder flexion (simulator theta)
thet2 = radians(0)
# sholder pronation
thet3 = radians(0)
# elbow

# neck to head
l0 = 2

T_id = np.matrix(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)

# T_b0[0:3, 0:3] = R.from_matrix(T_b0[0:3, 0:3]).as_matrix()

T_b1 = Ti(0, (np.pi / 2), 0, thet1 + (np.pi / 2))
# T_b1[0:3, 0:3] = R.from_matrix(T_b1[0:3, 0:3]).as_matrix()

T_12 = Ti(0, np.pi / 2, 0, thet2 + (3 * np.pi / 2))
# T_12[0:3, 0:3] = R.from_matrix(T_12[0:3, 0:3]).as_matrix()

T_23 = Ti(0, -np.pi / 2, 0, thet3 + np.pi)
# T_23[0:3, 0:3] = R.from_matrix(T_23[0:3, 0:3]).as_matrix()

T_3h = Ti(l0, 0, 0, 0)
# T_3h[0:3, 0:3] = R.from_matrix(T_3h[0:3, 0:3]).as_matrix()


# neck length
dx = -3
#
dy = 0
# z dist from shoulder height to eyes
dz = 0.25

# T3h = np.matrix(
#     [
#         [1, 0, 0, dx],
#         [0, 1, 0, dy],
#         [0, 0, 1, dz],
#         [0, 0, 0, 1],
#     ]
# )


# T_ke[0:3, 0:3] = R.from_matrix(T_ke[0:3, 0:3]).as_matrix()
# # or
# T_k_e = T_k0 @ T_01 @ T_12 @ T_23

# print(T_ke)
# print(T_k_e)
# print(T_ke == T_k_e)


# tm = TransformManager()
# tm.add_transform("b", "0", T_b0)
# tm.add_transform("0", "1", T_01)

# tm.add_transform("1", "2", T_12)
# tm.add_transform("2", "3", T_23)
# tm.add_transform("3", "4", T_34)
# tm.add_transform("4", "e", T_4e)

# tm.add_transform("k", "e", T_ke)


plt.figure(figsize=(9, 9))

ax = make_3d_axis(2, 121)
ax = show_Ti(ax, T_id, "gold", "b")
ax = show_Ti(ax, T_b1, "green", 1)
ax = show_Ti(ax, (T_12 @ T_b1), "crimson", 2)
# ax = show_Ti(ax, T_23 @ T_12 @ T_b1, "blue", 3)
# ax = show_Ti(ax, T_3h @ T_23 @ T_12 @ T_b1, "gold", "H")
ax.view_init(40, 40)


plt.show()
