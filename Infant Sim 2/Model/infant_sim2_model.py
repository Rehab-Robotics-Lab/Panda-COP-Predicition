import numpy as np
import math
from numpy import radians as radians
from scipy.spatial.transform import Rotation as R


class infant_sim2:
    def __init__(self, theta, arm, leg):
        ##INPUT
        self.theta = radians(theta)
        # time step between input angles
        dt = 0.5

        # angular velocity (will probably need to specify axis later)
        self.theta_d = np.gradient(theta, dt, edge_order=1, axis=1)
        # angular acceleration
        self.theta_dd = np.gradient(theta, dt, edge_order=2, axis=1)

        ##CONSTANTS
        # axis of rotation (always like this b/c axis of rot is always Z)
        # self.z0 = np.matix([[0], [0], [1]])

        ##INFANT PARAMETERS
        # fromat =[length,mass, Ix,Iy,Iz]
        # upper and lower arm paramters [2x5]
        self.arm = arm
        # upper and lower leg paramters [2x5]
        self.leg = leg

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
        R = np.zeros((3, 3, 3))
        P = np.zeros((3, 3))

        # T_k0 = Ti(0, 0, l0, 0)
        # T_k0[0:3, 0:3] = R.from_matrix(T_k0[0:3, 0:3]).as_matrix()

        # setting refrence frame at base of joint
        l1 = 0
        # setting l2 and le
        l2, le = self.arm[0, 0], self.arm[1, 0]

        # specifying the limb we're looking at and making sure to only use those angles
        idx = self.limb_dict[DOF]
        thet = self.theta[:, :, idx]

        T01 = self.Ti(0, l1, 0, thet[0, i])
        # T_01[0:3, 0:3] = R.from_matrix(T_01[0:3, 0:3]).as_matrix()

        T12 = self.Ti(np.pi / 2, 0, 0, thet[1, i])
        # T_12[0:3, 0:3] = R.from_matrix(T_12[0:3, 0:3]).as_matrix()

        T23 = self.Ti(thet[2, i], 0, 0, 0)
        # T_23[0:3, 0:3] = R.from_matrix(T_23[0:3, 0:3]).as_matrix()
        # transfromation matrix from base with all shoulder angles
        T03 = T01 @ T12 @ T23
        R[:, :, 0] = T03[0:3, 0:3]
        P[:, [0]] = T03[0:3, 3]

        T34 = self.Ti(0, l2, 0, thet[3, i])
        # T_34[0:3, 0:3] = R.from_matrix(T_34[0:3, 0:3]).as_matrix()
        R[:, :, 1] = T34[0:3, 0:3]
        P[:, [1]] = T34[0:3, 3]

        T4e = self.Ti(0, le, 0, 0)
        # T_4e[0:3, 0:3] = R.from_matrix(T_4e[0:3, 0:3]).as_matrix()
        R[:, :, 2] = T4e[0:3, 0:3]
        P[:, [2]] = T4e[0:3, 3]

        # return 3 R and P matrices
        # print(R)
        # print(P)

        # # return 5 R and P matrices
        # R = np.ndarray(
        #     [T01[0:3, 0:3], T12[0:3, 0:3], T23[0:3, 0:3], T34[0:3, 0:3], T4e[0:3, 0:3]]
        # )
        # P = np.ndarray(
        #     [T01[0:3, 3], T12[0:3, 3], T23[0:3, 3], T34[0:3, 3], T4e[0:3, 3]]
        # )

        return R, P

    def forward(self, R, P, limb, vel, accel):
        # Iterate over limbs (3 or 2)
        # initialize w0,w0_d, v_0d
        w0 = np.zeros([3, 1])
        w0_d = np.zeros([3, 1])
        v0_d = np.zeros([3, 1])
        v0_d[2] = -9.81

        iters = 2
        F = np.zeros((3, iters))
        N = np.zeros((3, iters))

        # # initializing paramteres from refrence to first frame
        # # limb mass
        # m1 = 0
        # # finding r to COM as half of limb length
        # r1 = np.matrix([[0], [0], [0]])
        # # inertia tensor(might be wrong)
        # I = np.diag((1, 1, 1))

        for i in range(iters):
            # limb mass
            m1 = limb[i, 1]
            # finding r to COM as half of limb length
            r1 = np.matrix([[limb[i, 0] / 2], [0], [0]])
            # inertia tensor
            I = np.diag((limb[i, 2], limb[i, 3], limb[i, 4]))

            # thetas
            theta1_d = vel[:, i]
            theta1_dd = accel[:, i]

            # R and P for given iteration
            R1 = R[i, :, :]
            P1 = P[i, :]

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
        iters = 2

        # initializing f1 and n1
        f1 = np.zeros([3, 1])
        n1 = np.zeros([3, 1])
        for i in range(iters):
            j = -1 - i

            F0 = F[:, [j]]
            P1 = P[j]
            N0 = N[:, [j]]

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
            theta1_d = np.matrix(
                [
                    [self.theta_d[0, i, idx]],
                    [self.theta_d[1, i, idx]],
                    [self.theta_d[2, i, idx]],
                ]
            )
            theta1_dd = np.matrix(
                [
                    [self.theta_dd[0, i, idx]],
                    [self.theta_dd[1, i, idx]],
                    [self.theta_dd[2, i, idx]],
                ]
            )

            # theta dot and theta double dot at elbow
            theta2_d = np.matrix([[0], [0], [self.theta_d[3, i, idx]]])
            theta2_dd = np.matrix([[0], [0], [self.theta_dd[3, i, idx]]])

            # theta dot and theta double dot at wrist (all zero because we assume no rotation at wrist)
            theta3_d = np.matrix([[0], [0], [0]])
            theta3_dd = np.matrix([[0], [0], [0]])

            ang_vel = np.hstack((theta1_d, theta2_d, theta3_d))
            ang_accel = np.hstack((theta1_dd, theta2_dd, theta3_dd))

            # Forward kinematics
            R, P = self.FK(i, limb, DOF)
            # foward iteration
            F, N = self.forward(R, P, limb, ang_vel, ang_accel)
            self.backward(R, P, limb, F, N)


# test angles for arm
