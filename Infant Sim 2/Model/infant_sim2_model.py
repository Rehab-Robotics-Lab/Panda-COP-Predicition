import numpy as np
import math
from numpy import radians as radians
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transform_manager import TransformManager


class infant_sim2:
    def __init__(self, theta, arm, leg):
        ##INPUT
        self.theta = radians(theta)
        # time step between input angles
        dt = 0.5

        # angular velocity (will probably need to specify axis later)
        self.theta_d = np.gradient(self.theta, dt, edge_order=1, axis=1)
        # angular acceleration
        self.theta_dd = np.gradient(self.theta_d, dt, edge_order=1, axis=1)

        ##CONSTANTS
        # axis of rotation (always like this b/c axis of rot is always Z)
        # self.z0 = np.matix([[0], [0], [1]])

        ##INFANT PARAMETERS
        # fromat =[length,mass, Ix,Iy,Iz]
        # upper and lower arm paramters [2x5]
        self.arm = arm
        # upper and lower leg paramters [2x5]
        self.leg = leg

        ## iteration ready version of interatia value
        # Creating base matrix (3=>inertia paramteres, 4=>number of iterations, 2=>arms and legs)
        I = np.ones((3, 4, 2))
        # upper arm
        I[:, [2], 0] = np.matrix([[arm[0, 2]], [arm[0, 3]], [arm[0, 4]]])
        # lower arm
        I[:, [3], 0] = np.matrix([[arm[1, 2]], [arm[1, 3]], [arm[1, 4]]])

        # upper leg
        I[:, [2], 1] = np.matrix([[leg[0, 2]], [leg[0, 3]], [leg[0, 4]]])
        # lower leg
        I[:, [3], 1] = np.matrix([[leg[1, 2]], [leg[1, 3]], [leg[1, 4]]])

        self.I = I

        # dictionary corresponding limb names to index
        self.limb_dict = {"larm": 0, "rarm": 1, "lleg": 2, "rleg": 3}

    #   def __str__(self):
    #     return f"{self.name}({self.age})"

    def Ti(self, alpha0, ai, di, thetai):
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

    # forward kinematics input iteration and limb angle
    def FK(self, i, limb, DOF):
        R = np.zeros((3, 3, 5))
        P = np.zeros((3, 5))

        # T_k0 = Ti(0, 0, l0, 0)
        # T_k0[0:3, 0:3] = R.from_matrix(T_k0[0:3, 0:3]).as_matrix()

        # setting refrence frame at base of joint
        l1 = 0
        # setting l2 and le
        l2, le = self.arm[0, 0], self.arm[1, 0]

        # specifying the limb we're looking at and making sure to only use those angles
        idx = self.limb_dict[DOF]
        thet = self.theta[:, :, idx]

        self.T01 = self.Ti(0, l1, 0, thet[0, i])
        R[:, :, 0] = self.T01[0:3, 0:3]
        P[:, [0]] = self.T01[0:3, 3]
        # T_01[0:3, 0:3] = R.from_matrix(T_01[0:3, 0:3]).as_matrix()

        self.T12 = self.Ti(np.pi / 2, 0, 0, thet[1, i])
        R[:, :, 1] = self.T12[0:3, 0:3]
        P[:, [1]] = self.T12[0:3, 3]
        # T_12[0:3, 0:3] = R.from_matrix(T_12[0:3, 0:3]).as_matrix()

        self.T23 = self.Ti(thet[2, i], 0, 0, 0)
        R[:, :, 2] = self.T23[0:3, 0:3]
        P[:, [2]] = self.T23[0:3, 3]
        # T_23[0:3, 0:3] = R.from_matrix(T_23[0:3, 0:3]).as_matrix()

        # # transfromation matrix from base with all shoulder angles
        # self.T03 = self.T01 @ self.T12 @ self.T23
        # R[:, :, 0] = self.T03[0:3, 0:3]
        # P[:, [0]] = self.T03[0:3, 3]

        self.T34 = self.Ti(0, l2, 0, thet[3, i])
        # T_34[0:3, 0:3] = R.from_matrix(T_34[0:3, 0:3]).as_matrix()
        R[:, :, 3] = self.T34[0:3, 0:3]
        P[:, [3]] = self.T34[0:3, 3]

        self.T4e = self.Ti(0, le, 0, 0)
        # T_4e[0:3, 0:3] = R.from_matrix(T_4e[0:3, 0:3]).as_matrix()
        R[:, :, 4] = self.T4e[0:3, 0:3]
        P[:, [4]] = self.T4e[0:3, 3]

        # for j in range(5):
        #     print("frame: ", j)
        #     print(R[:, :, j])
        #     print(P[:, j])

        return R, P

    def vis_FK(self):
        T01 = self.T01
        T01[0:3, 0:3] = R.from_matrix(T01[0:3, 0:3]).as_matrix()
        T12 = self.T12
        T12[0:3, 0:3] = R.from_matrix(T12[0:3, 0:3]).as_matrix()
        T23 = self.T23
        T23[0:3, 0:3] = R.from_matrix(T23[0:3, 0:3]).as_matrix()
        T34 = self.T34
        print(T34)
        T34[0:3, 0:3] = R.from_matrix(T34[0:3, 0:3]).as_matrix()
        T4e = self.T4e
        T4e[0:3, 0:3] = R.from_matrix(T4e[0:3, 0:3]).as_matrix()

        # T03 = self.T03
        # T03[0:3, 0:3] = R.from_matrix(T03[0:3, 0:3]).as_matrix()

        tm = TransformManager()
        tm.add_transform("0", "1", T01)
        tm.add_transform("1", "2", T12)
        tm.add_transform("2", "3", T23)

        # tm.add_transform("0", "3", T03)

        tm.add_transform("3", "4", T34)
        tm.add_transform("4", "e", T4e)

        plt.figure(figsize=(8, 12))

        ax = make_3d_axis(2, 121)
        ax = tm.plot_frames_in("0", ax=ax, alpha=0.6)
        ax.view_init(30, 20)

        plt.show()

    def forward(self, R, P, limb, thet_d, thet_dd):
        # Iterate over limbs (3 or 2)
        # initialize w0,w0_d, v_0d
        w0 = np.zeros([3, 1])
        w0_d = np.zeros([3, 1])
        v0_d = np.zeros([3, 1])
        v0_d[2] = -9.81

        z0 = np.matrix([[0], [0], [1]])

        iters = 4
        F = np.zeros((3, iters))
        N = np.zeros((3, iters))

        # # initializing paramteres from refrence to first frame
        # # limb mass
        # m1 = 0
        # # finding r to COM as half of limb length
        # r1 = np.matrix([[0], [0], [0]])
        # # inertia tensor(might be wrong)
        # I = np.diag((1, 1, 1))
        m = np.array([0, 0, limb[0, 1], limb[1, 1]])

        if np.all(limb == self.arm):
            idx = 0
        elif np.all(limb == self.leg):
            idx = 1

        for i in range(iters):
            # R and P for given iteration
            R1 = R[:, :, i]
            P1 = P[:, i]
            # needed to find r1/p1c
            P2c = P[:, i + 1] / 2

            # limb mass
            m1 = m[i]
            # finding r to COM as half of limb length
            r1 = P1 + P2c
            # inertia tensor
            I = np.diag((self.I[:, i, idx]))

            # thetas
            theta1_d = thet_d[i] * z0
            theta1_dd = thet_dd[i] * z0

            ## ANGULAR
            # angular velocity
            w1 = R1.T @ w0 + (theta1_d)
            # angular accelearation
            w1_d = R1.T @ w0_d + np.cross((R1.T @ w0), theta1_d, axis=0) + theta1_dd

            ## LINEAR
            # linear accelearation
            v1_d = R1.T @ (
                np.cross(w0_d, P1, axis=0)
                + np.cross(w0, np.cross(w0, P1, axis=0), axis=0)
                + v0_d
            )
            # linear acceleration relative to COM
            vc1_d = (
                np.cross(w1, r1, axis=0)
                + np.cross(w1, np.cross(w1, r1, axis=0), axis=0)
                + v1_d
            )

            ## FORCES
            F1 = m1 * vc1_d
            # moments
            N1 = I @ w1_d + np.cross(w1, (I @ w1), axis=0)

            # add F1 and N1 to F and N
            F[:, [i]] = F1
            N[:, [i]] = N1

            w0 = w1
            w0_d = w1_d
            v0_d = v1_d

        print("F= ", F)
        print("N= ", N)

        return F, N

    def backward(self, R, P, limb, F, N):
        iters = 4

        # initializing f1 and n1
        f1 = np.zeros([3, 1])
        n1 = np.zeros([3, 1])

        P2c = np.zeros((3, 1))

        for i in range(iters):
            j = -1 - i

            F0 = F[:, [j]]
            P1 = P[:, [j]]
            N0 = N[:, [j]]

            P2c = P[:, i + 1] / 2
            # finding r to COM as half of limb length
            r1 = P1 + P2c

            r0 = np.matrix([[limb[j, 0] / 2], [0], [0]])

            # force propagation
            f0 = R[j] @ f1 + F0
            # accumlatve torque at joints
            n0 = (
                N0
                + R[j] @ n1
                + np.cross(r0, F0, axis=0)
                + np.cross(P1, (R[j] @ f1), axis=0)
            )
            # # torque at joint
            # T0 = n0

            # reassigning f1 and n1 for next iteration
            f1 = f0
            n1 = n0

        print("f= ", f0)
        print("T= ", n0)
        # return f, T

    # iterates through time
    def inv_dynamics(self, limb, DOF):
        idx = self.limb_dict[DOF]
        # limb = np.vstack((limb, np.zeros((1, 5))))
        for i in range(1):

            # theta dot and theta double dot at shoulder
            thet_d = self.theta_d[:, i, idx]
            thet_dd = self.theta_dd[:, i, idx]

            # Forward kinematics
            R, P = self.FK(i, limb, DOF)

            self.vis_FK()

            quit()

            # foward iteration
            F, N = self.forward(R, P, limb, thet_d, thet_dd)

            quit
            self.backward(R, P, limb, F, N)


# test angles for arm
