import numpy as np
import math
from numpy import radians as radians
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transform_manager import TransformManager


class inv_dynamics:
    # mass=[2X1], length=[2x1], I=[2x3], theta=[4,t]
    def __init__(self, theta, length, mass, I, dt):
        ##INPUT
        self.theta = radians(theta)

        # angular velocity (will probably need to specify axis later)
        self.theta_d = np.gradient(self.theta, dt, axis=1)
        # angular acceleration
        self.theta_dd = np.gradient(self.theta_d, dt, axis=1)

        ##INFANT PARAMETERS
        # upper and lower arm paramters [2x5]
        self.length = length
        self.mass = mass

        self.I = I

        # dictionary corresponding limb names to index
        # self.limb_dict = {"larm": 0, "rarm": 1, "lleg": 2, "rleg": 3}

    #   def __str__(self):
    #     return f"{self.name}({self.age})"

    def Ti(self, ai, alpha0, di, thetai):
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
    def FK(self, i):
        # setting l1 and l2
        l1, l2 = self.length[0], self.length[1]

        # specifying the iteration of angles we're looking at
        thet = self.theta[:, i]

        self.T01 = self.Ti(0, np.pi / 2, 0, thet[0] + np.pi / 2)
        self.R01 = self.Ri(self.T01)

        self.T12 = self.Ti(0, np.pi / 2, 0, thet[1] + (3 * np.pi / 2))
        self.R12 = self.Ri(self.T12)

        self.T23 = self.Ti(0, -np.pi / 2, 0, np.pi + thet[2])
        self.R23 = self.Ri(self.T23)

        self.T34 = self.Ti(l1, np.pi / 2, 0, thet[3])
        self.R34 = self.Ri(self.T34)
        self.P34 = self.Pi(self.T34)

        self.T4e = self.Ti(l2, 0, 0, 0)
        self.R4e = self.Ri(self.T4e)
        self.P4e = self.Pi(self.T4e)

        # finding positions of shoulder to elbow wrt to first few frames
        self.P24 = self.R23 @ self.P34
        self.P14 = self.R12 @ self.R23 @ self.P34
        self.P04 = self.R01 @ self.R12 @ self.R23 @ self.P34

    def Ri(self, T):
        R = T[0:3, 0:3]
        return R

    def Pi(self, T):
        P = T[0:3, 3]
        return P

    def vis_FK(self):
        T01 = self.T01
        T01[0:3, 0:3] = R.from_matrix(T01[0:3, 0:3]).as_matrix()
        T12 = self.T12
        T12[0:3, 0:3] = R.from_matrix(T12[0:3, 0:3]).as_matrix()
        T23 = self.T23
        T23[0:3, 0:3] = R.from_matrix(T23[0:3, 0:3]).as_matrix()
        T34 = self.T34
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

    def ID(self, i):
        ###### FORWARD ITERATION
        w0 = np.zeros([3, 1])
        w0_d = np.zeros([3, 1])
        v0_d = np.zeros([3, 1])
        v0_d[2] = -9.81

        z0 = np.matrix([[0], [0], [1]])

        # naming angular values/derivs for iteration (for readability)
        thet1 = self.theta[0, i]
        thet2 = self.theta[1, i]
        thet3 = self.theta[2, i]
        thet4 = self.theta[3, i]
        # theta dot
        thet_d1 = self.theta_d[0, i]
        thet_d2 = self.theta_d[1, i]
        thet_d3 = self.theta_d[2, i]
        thet_d4 = self.theta_d[3, i]
        # theta double dot
        thet_dd1 = self.theta_dd[0, i]
        thet_dd2 = self.theta_dd[1, i]
        thet_dd3 = self.theta_dd[2, i]
        thet_dd4 = self.theta_dd[3, i]

        # F = np.zeros((3, iters))
        # N = np.zeros((3, iters))

        # # initializing paramteres from refrence to first frame

        ## ANGULAR
        # angular velocity
        w1 = self.R01.T @ w0 + z0 * thet_d1
        w2 = self.R12.T @ w1 + z0 * thet_d2
        w3 = self.R23.T @ w2 + z0 * thet_d3
        w4 = self.R34.T @ w3 + z0 * thet_d4

        # angular accelearation
        w1_d = (self.R01.T @ w0_d) + np.cross(self.R01.T @ w0, z0 * thet_d1, axis=0) + (z0 * thet_dd1)
        w2_d = (self.R12.T @ w1_d) + np.cross(self.R12.T @ w1, z0 * thet_d2, axis=0) + (z0 * thet_dd2)
        w3_d = (self.R23.T @ w2_d) + np.cross(self.R23.T @ w2, z0 * thet_d3, axis=0) + (z0 * thet_dd3)
        w4_d = (self.R34.T @ w3_d) + np.cross(self.R34.T @ w3, z0 * thet_d4, axis=0) + (z0 * thet_dd2)
        ## LINEAR

        # linear accelearation
        v1_d = self.R01.T @ (
            np.cross(w0_d, self.P04, axis=0) + np.cross(w0, np.cross(w0, self.P04, axis=0), axis=0) + v0_d
        )
        v2_d = self.R12.T @ (
            np.cross(w1_d, self.P14, axis=0) + np.cross(w1, np.cross(w1, self.P14, axis=0), axis=0) + v1_d
        )
        v3_d = self.R23.T @ (
            np.cross(w2_d, self.P24, axis=0) + np.cross(w2, np.cross(w2, self.P24, axis=0), axis=0) + v2_d
        )
        v4_d = self.R34.T @ (
            np.cross(w3_d, self.P34, axis=0) + np.cross(w3, np.cross(w1, self.P34, axis=0), axis=0) + v3_d
        )

        # linear acceleration relative to COM
        r1 = self.P14 / 2
        vc1_d = np.cross(w1_d, r1, axis=0) + np.cross(w1, np.cross(w1, r1, axis=0), axis=0) + v1_d
        r2 = self.P24 / 2
        vc2_d = np.cross(w2_d, r2, axis=0) + np.cross(w2, np.cross(w2, r2, axis=0), axis=0) + v2_d
        r3 = self.P34 / 2
        vc3_d = np.cross(w3_d, r3, axis=0) + np.cross(w3, np.cross(w3, r3, axis=0), axis=0) + v3_d
        r4 = self.P4e / 2
        vc4_d = np.cross(w4_d, r4, axis=0) + np.cross(w4, np.cross(w4, r4, axis=0), axis=0) + v4_d

        ## FORCES
        # froces from shoulder frames
        F3 = self.mass[0] * vc3_d
        # froces from ekbow frame
        F4 = self.mass[1] * vc4_d

        ##MOMENTS
        I3 = np.diag(np.ravel(self.I[0, :]))
        N3 = I3 @ w3_d + np.cross(w3, (I3 @ w3), axis=0)
        I4 = np.diag(np.ravel(self.I[1, :]))
        N4 = I4 @ w4_d + np.cross(w4, (I4 @ w4), axis=0)

        ###### BACKWARDS ITERATION
        # ACCUMULATED FORCES at joints
        f4 = F4
        f3 = self.R34 @ f4 + F3

        # ACCUMULATED TORQUE at joints
        n4 = N4
        n3 = N3 + self.R34 @ n4 + np.cross(r3, F3, axis=0) + np.cross(self.P34, (self.R34 @ f4), axis=0)

        # torque from shoulder flextion/extension
        T = self.R01 @ self.R12 @ self.R23 @ n3
        # forces transformed to base frame
        F = self.R01 @ self.R12 @ self.R23 @ f3

        return T, F

    # iterates through time
    def calc(self):
        a, n = np.shape(self.theta)

        T = np.zeros((3, n))
        F = np.zeros((3, n))

        for i in range(n):
            # Forward kinematics
            self.FK(i)

            # inverse dynamics
            T[:, [i]], F[:, [i]] = self.ID(i)

        return T, F


# test angles for arm
