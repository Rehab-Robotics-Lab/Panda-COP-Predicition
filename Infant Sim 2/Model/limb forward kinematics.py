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

# sholder abduction
thet1 = radians(0)
# sholder flexion (simulator theta)
thet2 = radians(0)
# sholder pronation
thet3 = radians(0)
# elbow
thet4 = radians(30)

# camera to cheast
l0 = 0
# chest to shoulder
l1 = 1
# shoulder to elbow
l2 = 2
# elbow to wrist
le = 2

r11 = (
    (math.cos(thet1) * math.cos(thet2) * math.cos(thet4))
    - (math.cos(thet1) * math.sin(thet2) * math.cos(thet3) * math.sin(thet4))
    + (math.sin(thet1) * math.sin(thet3) * math.sin(thet4))
)
r12 = (
    (-math.cos(thet1) * math.cos(thet2) * math.sin(thet4))
    - (math.cos(thet1) * math.sin(thet2) * math.cos(thet3) * math.cos(thet4))
    + (math.sin(thet1) * math.sin(thet3) * math.cos(thet4))
)
r13 = (math.cos(thet1) * math.sin(thet2) * math.sin(thet3)) + (
    math.sin(thet1) * math.cos(thet3)
)

r21 = (
    (math.sin(thet1) * math.cos(thet2) * math.cos(thet4))
    - (math.sin(thet1) * math.sin(thet2) * math.cos(thet3) * math.sin(thet4))
    - (math.cos(thet1) * math.sin(thet3) * math.sin(thet4))
)
r22 = (
    (-math.sin(thet1) * math.cos(thet2) * math.sin(thet4))
    - (math.sin(thet1) * math.sin(thet2) * math.cos(thet3) * math.cos(thet4))
    - (math.cos(thet1) * math.sin(thet3) * math.cos(thet4))
)
r23 = (math.sin(thet1) * math.sin(thet2) * math.sin(thet3)) - (
    math.cos(thet1) * math.cos(thet3)
)

r31 = (math.sin(thet2) * math.cos(thet4)) + (
    math.cos(thet2) * math.cos(thet3) * math.sin(thet4)
)
r32 = (-math.sin(thet2) * math.sin(thet4)) + (
    math.cos(thet2) * math.cos(thet3) * math.cos(thet4)
)
r33 = -math.cos(thet2) * math.sin(thet3)


T_ke = np.matrix(
    [
        [r11, r12, r13, le * r11 + l2 * math.cos(thet1) * math.cos(thet2) + l1],
        [r21, r22, r23, le * r21 + l2 * math.sin(thet1) * math.cos(thet2)],
        [r31, r32, r33, le * r31 + l2 * math.sin(thet2) + l0],
        [0, 0, 0, 1],
    ]
)


def Ti(alpha0, ai, di, thetai):
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


def forward(
    m1,
    R1,
    p1,
    w0,
    w0_d,
    v0_d,
    theta1_d,
    theta1_dd,
    Ixx,
    Iyy,
    Izz,
):
    # PARAMETERS
    # axis of rotation (always like this b/c axis of rot is always Z)
    z0 = np.matix([[0], [0], [1]])
    # position of COM realtive to link
    r1 = np.matix([[p1[1] / 2], [0], [0]])
    # inertia tensor
    I = np.diag(Ixx, Iyy, Izz)

    ## ANGULAR
    # angular velocity
    w1 = R1.T @ w0 + (theta1_d * z0)
    # angular accelearation
    w1_d = R1.T @ w0_d + np.cross((R1.T @ w0), (theta1_d * z0)) + (theta1_dd @ z0)

    ## LINEAR
    # linear accelearation
    v1_d = R1.T @ (np.cross(w0_d, p1) + np.cross(w0, np.cross(w0, p1)) + v0_d)
    # linear acceleration relative to COM
    vc1_d = np.cross(w1, r1) + np.cross(w1, np.cross(w1, r1)) + v1_d

    ## FORCES
    F1 = m1 * vc1_d
    # moments
    N1 = I @ w1_d + np.cross(w1, (I @ w1))

    return F1, N1


def backward(R1, f1, F0, N0, n1, r0, p1, z0):
    # joint force
    f0 = R1 @ f1 + F0
    # joint torque
    n0 = N0 + R1 @ n1 + np.cross(r0, F0) + np.cross(p1, R1 @ f1)
    # joint torque at axis of rotation
    T0 = n0.T @ z0


# make global/tensor (R,m,p)


# generic = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


T_k0 = Ti(0, 0, l0, 0)
T_k0[0:3, 0:3] = R.from_matrix(T_k0[0:3, 0:3]).as_matrix()

T_01 = Ti(0, l1, 0, thet1)
T_01[0:3, 0:3] = R.from_matrix(T_01[0:3, 0:3]).as_matrix()

T_12 = Ti(np.pi / 2, 0, 0, thet2)
T_12[0:3, 0:3] = R.from_matrix(T_12[0:3, 0:3]).as_matrix()

T_23 = Ti(thet3, 0, 0, 0)
T_23[0:3, 0:3] = R.from_matrix(T_23[0:3, 0:3]).as_matrix()

T_34 = Ti(0, l2, 0, thet4)
T_34[0:3, 0:3] = R.from_matrix(T_34[0:3, 0:3]).as_matrix()

T_4e = Ti(0, le, 0, 0)
T_4e[0:3, 0:3] = R.from_matrix(T_4e[0:3, 0:3]).as_matrix()


T_ke[0:3, 0:3] = R.from_matrix(T_ke[0:3, 0:3]).as_matrix()
# or
T_k_e = T_k0 @ T_01 @ T_12 @ T_23 @ T_34 @ T_4e

# print(T_ke)
# print(T_k_e)
# print(T_ke == T_k_e)


tm = TransformManager()
tm.add_transform("k", "0", T_k0)
tm.add_transform("0", "1", T_01)
tm.add_transform("1", "2", T_12)
tm.add_transform("2", "3", T_23)
# tm.add_transform("3", "4", T_34)
# tm.add_transform("4", "e", T_4e)

# tm.add_transform("k", "e", T_ke)


plt.figure(figsize=(8, 12))

ax = make_3d_axis(2, 121)
ax = tm.plot_frames_in("k", ax=ax, alpha=0.6)
ax.view_init(30, 20)


plt.show()
