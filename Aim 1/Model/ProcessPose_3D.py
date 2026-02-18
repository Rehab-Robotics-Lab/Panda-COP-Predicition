import numpy as np
import math
import scipy
from scipy.spatial.transform import Rotation as R
import pandas as pd
from numpy import radians as radians
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


class processpose:
    def __init__(self, file):

        # df = pd.read_pickle(file)
        df = pd.read_csv(file)

        # print(df)

        df["x"] = df["x"].astype(float)
        # df["y"] = df["y"].astype(float)
        df["y"] = df["y"].astype(float)

        frames = int(np.max(df.frame))
        self.frames = frames

        X = np.zeros((frames + 1, 18))
        Y = np.zeros((frames + 1, 18))
        Z = np.zeros((frames + 1, 18))

        idx = np.zeros((frames + 1, 18))

        for p in range(18):
            # print(p)
            X[:, p] = df.x[df.part_idx == p].T
            Y[:, p] = df.y[df.part_idx == p].T
            Z[:, p] = df.z[df.part_idx == p].T
            idx[:, p] = p

        # Butterworth Filter

        # order = 3
        # fcut = 5
        # b, a = scipy.signal.butter(order, fcut, fs=60)

        # X = scipy.signal.filtfilt(b, a, X)
        # Y = scipy.signal.filtfilt(b, a, Y)
        # Z = scipy.signal.filtfilt(b, a, Z)

        # Sgolay Filter

        # win = 10
        # order = 5

        # X = scipy.signal.savgol_filter(X, win, order, axis=0)
        # Y = scipy.signal.savgol_filter(Y, win, order, axis=0)
        # Z = scipy.signal.savgol_filter(Z, win, order, axis=0)

        # creating global variables
        win = 5

        X_init, Y_init, Z_init = np.mean(X[0:win, 1]), np.mean(Y[0:win, 1]), np.mean(Z[0:win, 1])

        self.X = X - X_init
        self.Y = Y - Y_init
        self.Z = Z
        self.idx = idx

    # def adjust_rleg_len(self):
    #     rk = np.matrix([self.X[:, 9], self.Y[:, 9], self.Z[:, 9]])
    #     rf = np.matrix([self.X[:, 10], self.Y[:, 10], self.Z[:, 10]])

    #     left_len = self.get_len(12, 13)
    #     right_len = self.get_len(9, 10)

    #     rf_proj = np.zeros_like(rf)

    #     # for i in range(self.frames + 1):
    #     #     Ts, Th, Tf, Te = self.IK_trunk(i)
    #     #     Ts_ra, Ts_re, Ts_la, Ts_le, Ts_rl, Ts_rk, Ts_ll, Ts_lk = self.IK_limbs(i, Ts, Th)
    #     #     # feet

    #     #     yfR = np.matrix([[0], [left_len[i]], [0], [1]])

    #     #     rf_proj[:, i] = (Ts_rk @ yfR)[0:3, 0]
    #     low_leg_norm = np.divide((rf - rk), right_len)
    #     rf_proj = np.multiply(low_leg_norm, left_len) + rk

    #     # print(np.shape(rf_proj))
    #     self.X[:, 10], self.Y[:, 10], self.Z[:, 10] = rf_proj[0, :], rf_proj[1, :], rf_proj[2, :]

    def get_len(self, i, j):
        p1 = np.matrix([self.X[:, i], self.Y[:, i], self.Z[:, i]])
        p2 = np.matrix([self.X[:, j], self.Y[:, j], self.Z[:, j]])

        L = np.linalg.norm(p1 - p2, axis=0)

        return L

    def plotskel(self, j, show_Ik=True, show_model=False):

        colors = np.matrix(
            [
                [255, 0, 0],
                [255, 170, 0],
                [255, 255, 0],
                [255, 85, 0],
                [170, 255, 0],
                [85, 255, 0],
                [0, 255, 0],
                [0, 255, 85],
                [0, 255, 170],
                [0, 255, 255],
                [0, 170, 255],
                [0, 85, 255],
                [0, 0, 255],
                [170, 0, 255],
                [255, 0, 255],
                [85, 0, 255],
                [85, 85, 255],
            ]
        )

        # colors = colors / 2

        # limbSeq = np.matrix(
        #     [[1, 0], [2, 1], [3, 2], [4, 0], [5, 4], [6, 5], [7, 0], [8, 7], [9, 8], [10, 0], [11, 10], [12, 11]]
        # )
        limbSeq = np.matrix(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [1, 5],
                [5, 6],
                [6, 7],
                [1, 8],
                [8, 9],
                [9, 10],
                [1, 11],
                [11, 12],
                [12, 13],
                [0, 14],
                [14, 16],
                [0, 15],
                [15, 17],
            ]
        )

        x = self.X[j, :]
        y = self.Y[j, :]
        z = self.Z[j, :]

        plt.cla()

        for p in range(0, 17):

            plt.plot(
                [x[limbSeq[p, 0]], x[limbSeq[p, 1]]],
                [y[limbSeq[p, 0]], y[limbSeq[p, 1]]],
                [z[limbSeq[p, 0]], z[limbSeq[p, 1]]],
                color=np.array(colors[p]) / 255,
                alpha=0.35,
            )
            plt.plot(
                [x[limbSeq[p, 0]], x[limbSeq[p, 1]]],
                [y[limbSeq[p, 0]], y[limbSeq[p, 1]]],
                [z[limbSeq[p, 0]], z[limbSeq[p, 1]]],
                "o",
                color=np.array(colors[p]) / 255,
                alpha=0.35,
            )

        if show_Ik == True:
            self.show_T(j)
        if show_model == True:
            self.model_overlay(j)

        s1 = "t3l= " + str(int(np.rad2deg(self.thet3_ll[0, j])))
        s2 = "t4l= " + str(int(np.rad2deg(self.thet4_ll[0, j])))

        s3 = "t3r= " + str(int(np.rad2deg(self.thet3_rl[0, j])))
        s4 = "t4r= " + str(int(np.rad2deg(self.thet4_rl[0, j])))
        # s2 = "s= " + str(round(j / 30))

        # self.ax.text(0.3, -0.3, 0, "%s" % (s1), size=20, zorder=1, color="k")
        # self.ax.text(0.3, -0.4, 0, "%s" % (s2), size=20, zorder=1, color="k")

        # self.ax.text(-0.6, -0.3, 0, "%s" % (s3), size=20, zorder=1, color="k")
        # self.ax.text(-0.6, -0.4, 0, "%s" % (s4), size=20, zorder=1, color="k")

        # plt.text(0, 0, s)
        self.ax.set_zlim3d(0, 0.6)
        self.ax.set_ylim3d(-0.4, 0.2)
        self.ax.set_xlim3d(-0.3, 0.3)
        plt.grid()

    def plotskel_loop(self, start=0):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")

        frms = np.linspace(0, self.frames, self.frames + 1, dtype=int)
        # print(frms)

        ani = animation.FuncAnimation(self.fig, self.plotskel, frames=frms[start:], interval=1)
        plt.show()

    def arm_ang(self, k):
        ls = np.matrix([self.X[:, k], self.Y[:, k], self.Z[:, k]])
        le = np.matrix([self.X[:, k + 1], self.Y[:, k + 1], self.Z[:, k + 1]])
        lh = np.matrix([self.X[:, k + 2], self.Y[:, k + 2], self.Z[:, k + 2]])

        L1 = np.linalg.norm(le - ls, axis=0)
        L2 = np.linalg.norm(le - lh, axis=0)

        s1 = np.divide(le[0, :] - ls[0, :], L1)
        c1 = np.divide(ls[1, :] - le[1, :], L1)

        thet1 = np.arctan2(s1, c1)
        s1 = np.sin(thet1)
        c1 = np.cos(thet1)

        s2 = np.divide(le[2, :] - ls[2, :], L1)
        c2 = np.divide((ls[1, :] - le[1, :]), np.multiply(c1, L1))

        thet2 = np.arctan2(s2, c2)
        s2 = np.sin(thet2)
        c2 = np.cos(thet2)

        u = le - ls
        v = lh - le
        # q = lh - ls

        cross_product = np.zeros_like(u)
        dot_product = np.zeros_like(u)

        for i in range(np.shape(thet1)[1]):
            cross_product[:, i] = np.cross(u[:, i].T, v[:, i].T).T

        dot_product = np.squeeze(np.sum(np.multiply(u, v), axis=0))

        magnitude_u = np.linalg.norm(u, axis=0)
        magnitude_v = np.linalg.norm(v, axis=0)
        # magnitude_q = np.linalg.norm(q, axis=0)
        magnitude_cross = np.linalg.norm(cross_product, axis=0)

        s4 = np.divide(magnitude_cross, np.multiply(magnitude_u, magnitude_v))
        c4 = np.divide(dot_product, np.multiply(magnitude_u, magnitude_v))

        thet4 = np.arctan2(s4, c4)

        c4 = np.cos(thet4)
        s4 = np.sin(thet4)

        ##THETA 3
        X = np.divide(lh[0, :] - le[0, :], L2)
        Y = np.divide(lh[1, :] - le[1, :], L2)
        Z = np.divide(lh[2, :] - le[2, :], L2)

        s3 = np.divide(-(np.multiply(X, c1) + np.multiply(Y, s1)), s4)
        c3 = np.divide((np.multiply(c2, c4) + np.multiply(Y, c1) - np.multiply(X, s1)), np.multiply(s2, s4))
        # c3 = np.sqrt((np.ones_like(s3) - np.power(s3, 2)).T).T

        thet3 = np.arctan2(s3, c3)

        return thet1, thet2, thet3, thet4

    def leg_ang(self, k):
        ls = np.matrix([self.X[:, k], self.Y[:, k], self.Z[:, k]])
        le = np.matrix([self.X[:, k + 1], self.Y[:, k + 1], self.Z[:, k + 1]])
        lh = np.matrix([self.X[:, k + 2], self.Y[:, k + 2], self.Z[:, k + 2]])

        L1 = np.linalg.norm(le - ls, axis=0)
        L2 = np.linalg.norm(lh - le, axis=0)

        s1 = np.divide(le[0, :] - ls[0, :], L1)
        c1 = np.divide(ls[1, :] - le[1, :], L1)

        thet1 = np.arctan2(s1, c1)
        s1 = np.sin(thet1)
        c1 = np.cos(thet1)

        s2 = np.divide(le[2, :] - ls[2, :], L1)
        c2 = np.divide((ls[1, :] - le[1, :]), np.multiply(c1, L1))
        thet2 = np.arctan2(s2, c2)
        s2 = np.sin(thet2)
        c2 = np.cos(thet2)

        u = ls - le
        v = le - lh
        q = ls - lh

        cross_product = np.zeros_like(u)
        dot_product = np.zeros_like(u)

        for i in range(np.shape(thet1)[1]):
            cross_product[:, i] = np.cross(u[:, i].T, v[:, i].T).T

        dot_product = np.squeeze(np.sum(np.multiply(u, v), axis=0))

        magnitude_u = np.linalg.norm(u, axis=0)
        magnitude_v = np.linalg.norm(v, axis=0)
        magnitude_q = np.linalg.norm(q, axis=0)
        magnitude_cross = np.linalg.norm(cross_product, axis=0)

        # c4 = [
        #     np.divide(
        #         (np.power(L1, 2) + np.power(L2, 2) - np.power(magnitude_q, 2)),
        #         (2 * np.multiply(L1, L2)),
        #     )
        # ]

        s4 = np.divide(magnitude_cross, np.multiply(magnitude_u, magnitude_v))
        c4 = np.divide(dot_product, np.multiply(magnitude_u, magnitude_v))

        thet4 = np.arctan2(s4, c4)
        # thet4 = np.arccos(c4)
        thet4 = -(np.pi - thet4)
        # c4 = np.cos(thet4)
        # s4 = np.sin(thet4)

        ##THETA 3

        X = np.divide(lh[0, :] - le[0, :], L2)
        Y = np.divide(lh[1, :] - le[1, :], L2)
        Z = np.divide(lh[2, :] - le[2, :], L2)

        s3 = np.divide(-(np.multiply(X, c1) + np.multiply(Y, s1)), s4)
        c3 = np.divide((np.multiply(c2, c4) + np.multiply(Y, c1) - np.multiply(X, s1)), np.multiply(s2, s4))

        # c3 = np.divide((Z - np.multiply(s2, c4)), np.multiply(c2, s4))
        # s3 = np.divide(
        #     -(np.multiply(c1, c4) + np.multiply(Y, c2) - np.multiply(np.multiply(Z, c1), s2)),
        #     np.multiply(np.multiply(s1, c2), s4),
        # )

        # c3 = np.sqrt((np.ones_like(s3) - np.power(s3, 2)).T).T
        # s3 = np.sqrt((np.ones_like(c3) - np.power(c3, 2)).T).T
        thet3 = np.arctan2(s3, c3)

        return thet1, thet2, thet3, thet4

    def face_ang(self):
        reye = np.matrix([self.X[:, 14], self.Y[:, 14], self.Z[:, 14]])
        leye = np.matrix([self.X[:, 15], self.Y[:, 15], self.Z[:, 15]])
        nose = np.matrix([self.X[:, 0], self.Y[:, 0], self.Z[:, 0]])

        mid = np.add(leye, reye) * 0.5
        self.mid_face = mid

        s1 = mid[2, :] - nose[2, :]
        c1 = mid[1, :] - nose[1, :]

        thet1 = np.arctan2(s1, c1)
        c1 = np.cos(thet1)
        s1 = np.sin(thet1)
        self.thet1f = thet1

        c2 = np.divide((mid[1, :] - nose[1, :]), np.cos(thet1))
        s2 = nose[0, :] - mid[0, :]

        thet2 = np.arctan2(s2, c2)
        self.thet2f = thet2
        c2 = np.cos(thet2)
        s2 = np.sin(thet2)

        c3 = np.divide((leye[0, :] - mid[0, :]), np.cos(thet2))
        # t12 = np.multiply(np.divide(s1, c1), np.divide(s2, c2))
        t12 = np.multiply(np.tan(thet1), np.tan(thet2))
        s3 = np.multiply(t12, (leye[0, :] - mid[0, :])) + (np.divide((mid[2, :] - leye[2, :]), np.cos(thet2)))

        thet3 = np.arctan2(s3, c3)
        self.thet3f = thet3

    def head_ang(self):
        rear = np.matrix([self.X[:, 16], self.Y[:, 16], self.Z[:, 16]])
        lear = np.matrix([self.X[:, 17], self.Y[:, 17], self.Z[:, 17]])
        neck = np.matrix([self.X[:, 1], self.Y[:, 1], self.Z[:, 1]])

        mid = np.add(lear, rear) * 0.5
        self.mid_ear = mid

        s1 = mid[2, :] - neck[2, :]
        c1 = mid[1, :] - neck[1, :]

        thet1 = np.arctan2(s1, c1)
        c1 = np.cos(thet1)
        s1 = np.sin(thet1)
        self.thet1e = thet1

        c2 = np.divide((mid[1, :] - neck[1, :]), np.cos(thet1))
        s2 = neck[0, :] - mid[0, :]

        thet2 = np.arctan2(s2, c2)
        c2 = np.cos(thet2)
        s2 = np.sin(thet2)
        self.thet2e = thet2

        c3 = np.divide((lear[0, :] - mid[0, :]), np.cos(thet2))
        # t12 = np.multiply(np.divide(s1, c1), np.divide(s2, c2))
        t12 = np.multiply(np.tan(thet1), np.tan(thet2))
        s3 = np.multiply(t12, (lear[0, :] - mid[0, :])) + (np.divide((mid[2, :] - lear[2, :]), np.cos(thet2)))

        thet3 = np.arctan2(s3, c3)
        self.thet3e = thet3

    def shoudler_ang(self):

        rshoulder = np.matrix([self.X[:, 2], self.Y[:, 2], self.Z[:, 2]])
        lshoulder = np.matrix([self.X[:, 5], self.Y[:, 5], self.Z[:, 5]])

        mid = np.add(lshoulder, rshoulder) * 0.5
        self.mid_shoulder = mid

        ##ZY
        s1 = lshoulder[1, :] - mid[1, :]
        c1 = lshoulder[0, :] - mid[0, :]

        thet1 = np.arctan2(s1, c1)
        c1 = np.cos(thet1)
        s1 = np.sin(thet1)
        self.thet1s = thet1

        # c2 = np.divide((lshoulder[0, :] - mid[0, :]), c1)
        s2 = mid[2, :] - lshoulder[2, :]
        # c2 = lshoulder[0, :] - mid[0, :]
        # s2 = np.multiply((mid[2, :] - lshoulder[2, :]), np.cos(thet1))
        c2 = np.sqrt(np.ones_like(s2) - np.power(s2, 2))

        thet2 = np.arctan2(s2, c2)
        self.thet2s = thet2

    def hip_ang(self):
        rhip = np.matrix([self.X[:, 8], self.Y[:, 8], self.Z[:, 8]])
        lhip = np.matrix([self.X[:, 11], self.Y[:, 11], self.Z[:, 11]])
        neck = np.matrix([self.X[:, 1], self.Y[:, 1], self.Z[:, 1]])

        mid = np.add(lhip, rhip) * 0.5
        self.mid_hip = mid

        s1 = neck[2, :] - mid[2, :]
        c1 = neck[1, :] - mid[1, :]

        thet1 = np.arctan2(s1, c1)
        self.thet1h = thet1
        c1 = np.cos(thet1)
        s1 = np.sin(thet1)

        c2 = np.divide((neck[1, :] - mid[1, :]), c1)
        s2 = mid[0, :] - neck[0, :]

        thet2 = np.arctan2(s2, c2)
        c2 = np.cos(thet2)
        s2 = np.sin(thet2)
        self.thet2h = thet2

        c3 = np.divide((lhip[0, :] - mid[0, :]), np.cos(thet2))
        # t12 = np.multiply(np.divide(s1, c1), np.divide(s2, c2))
        t12 = np.multiply(np.tan(thet1), np.tan(thet2))
        s3 = np.multiply(t12, (lhip[0, :] - mid[0, :])) + (np.divide((mid[2, :] - lhip[2, :]), c2))

        thet3 = np.arctan2(s3, c3)
        self.thet3h = thet3

    def R_x(self, t1):

        R = np.matrix(
            [
                [1, 0, 0],
                [0, np.cos(t1), -np.sin(t1)],
                [0, np.sin(t1), np.cos(t1)],
            ]
        )

        return R

    def R_y(self, t1):

        R = np.matrix(
            [
                [np.cos(t1), 0, np.sin(t1)],
                [0, 1, 0],
                [-np.sin(t1), 0, np.cos(t1)],
            ]
        )

        return R

    def R_z(self, t1):

        R = np.matrix(
            [
                [np.cos(t1), -np.sin(t1), 0],
                [np.sin(t1), np.cos(t1), 0],
                [0, 0, 1],
            ]
        )

        return R

    def R_zy(self, t1, t2):

        R = np.matrix(
            [
                [np.cos(t1) * np.cos(t2), -np.sin(t1), np.cos(t1) * np.sin(t2)],
                [np.sin(t1) * np.cos(t2), np.cos(t1), np.sin(t1) * np.sin(t2)],
                [-np.sin(t2), 0, np.cos(t2)],
            ]
        )

        return R

    def R_yz(self, t1, t2):

        R = np.matrix(
            [
                [np.cos(t1) * np.cos(t2), -np.cos(t1) * np.sin(t2), np.sin(t1)],
                [np.sin(t2), np.cos(t2), 0],
                [-np.sin(t1) * np.cos(t2), np.sin(t1) * np.sin(t2), np.cos(t1)],
            ]
        )
        return R

    def R_xzy(self, t1, t2, t3):

        R = np.matrix(
            [
                [(np.cos(t2) * np.cos(t3)), -np.sin(t2), (np.cos(t2) * np.sin(t3))],
                [
                    ((np.cos(t1) * np.sin(t2) * np.cos(t3)) + (np.sin(t1) * np.sin(t3))),
                    (np.cos(t1) * np.cos(t2)),
                    ((np.cos(t1) * np.sin(t2) * np.sin(t3)) - (np.sin(t1) * np.cos(t3))),
                ],
                [
                    ((np.sin(t1) * np.sin(t2) * np.cos(t3)) - (np.cos(t1) * np.sin(t3))),
                    (np.sin(t1) * np.cos(t2)),
                    ((np.sin(t1) * np.sin(t2) * np.sin(t3)) + (np.cos(t1) * np.cos(t3))),
                ],
            ]
        )

        return R

    def R_zxy(self, t1, t2, t3):

        R = np.matrix(
            [
                [
                    (-(np.sin(t1) * np.sin(t2) * np.sin(t3)) + (np.cos(t1) * np.cos(t3))),
                    -(np.sin(t1) * np.cos(t2)),
                    ((np.sin(t1) * np.sin(t2) * np.cos(t3)) + (np.cos(t1) * np.sin(t3))),
                ],
                [
                    ((np.cos(t1) * np.sin(t2) * np.sin(t3)) + (np.sin(t1) * np.cos(t3))),
                    (np.cos(t1) * np.cos(t2)),
                    (-(np.cos(t1) * np.sin(t2) * np.cos(t3)) + (np.sin(t1) * np.sin(t3))),
                ],
                [-(np.cos(t2) * np.sin(t3)), np.sin(t2), (np.cos(t2) * np.cos(t3))],
            ]
        )

        return R

    def reproj_IK(self, axis, T, i, P1, P2, mid=0):
        if mid == 0:
            p1 = np.matrix([self.X[i, P1], self.Y[i, P1], self.Z[i, P1]])
            p2 = np.matrix([self.X[i, P2], self.Y[i, P2], self.Z[i, P2]])
        else:
            k1 = np.matrix([self.X[i, P1], self.Y[i, P1], self.Z[i, P1]])
            k2 = np.matrix([self.X[i, P2], self.Y[i, P2], self.Z[i, P2]])

            p1 = (k1 + k2) * 0.5
            p2 = k2

        # print(p1,p2)
        L = np.linalg.norm(p1 - p2, axis=1)[0]

        # print(L)

        if axis == "x":
            vec = np.matrix([[L], [0], [0], [1]])
        elif axis == "y":
            vec = np.matrix([[0], [L], [0], [1]])

        p2_proj = np.zeros_like(p2)
        p2_proj = (T @ vec)[0:3, 0]
        diff = np.linalg.norm(p2 - p2_proj.T)

        thresh = 0.01

        check = diff >= thresh

        return diff, check

    def check_IK_all(self):
        n = self.frames + 1
        diff_S = np.zeros((n))
        diff_H = np.zeros((n))
        diff_F = np.zeros((n))
        diff_Rf = np.zeros((n))
        diff_Rh = np.zeros((n))

        check_S = np.zeros((n))
        check_H = np.zeros((n))
        check_F = np.zeros((n))
        check_Rf = np.zeros((n))
        check_Rh = np.zeros((n))

        # print(np.shape(diff_F),np.shape)

        for i in range(n):
            Ts, Th, Tf, Te = self.IK_trunk(i)
            Ts_ra, Ts_re, Ts_la, Ts_le, Ts_rl, Ts_rk, Ts_ll, Ts_lk = self.IK_limbs(i, Ts, Th)

            diff_S[i], check_S[i] = self.reproj_IK(axis="x", T=Ts, i=i, P1=2, P2=5, mid=1)
            diff_H[i], check_H[i] = self.reproj_IK(axis="x", T=Th, i=i, P1=8, P2=11, mid=1)
            diff_F[i], check_F[i] = self.reproj_IK(axis="x", T=Tf, i=i, P1=14, P2=15, mid=1)
            diff_Rf[i], check_Rf[i] = self.reproj_IK(axis="y", T=Ts_rk, i=i, P1=3, P2=4, mid=0)

        print("Shoulders: ", sum(check_S))
        print("Hips: ", sum(check_H))
        print("Face: ", sum(check_F))
        print("RFoot: ", sum(diff_Rf))

        plt.plot(diff_S)
        plt.plot(diff_H)
        plt.plot(diff_F)
        plt.plot(diff_Rf)
        plt.legend(["Shoulders", "Hips", "Face", "RFoot"])
        plt.show()

    def check_IK_legs(self, k=2):
        lh = np.matrix([self.X[:, k + 3], self.Y[:, k + 3], self.Z[:, k + 3]])
        lk = np.matrix([self.X[:, k + 4], self.Y[:, k + 4], self.Z[:, k + 4]])
        lf = np.matrix([self.X[:, k + 5], self.Y[:, k + 5], self.Z[:, k + 5]])

        L1L = np.linalg.norm(lk - lh, axis=0)
        L2L = np.linalg.norm(lk - lf, axis=0)

        rh = np.matrix([self.X[:, k], self.Y[:, k], self.Z[:, k]])
        rk = np.matrix([self.X[:, k + 1], self.Y[:, k + 1], self.Z[:, k + 1]])
        rf = np.matrix([self.X[:, k + 2], self.Y[:, k + 2], self.Z[:, k + 2]])

        # print(np.shape(rf))

        L1R = np.linalg.norm(rk - rh, axis=0)
        L2R = np.linalg.norm(rk - rf, axis=0)

        # origin = np.matrix([[0], [0], [0], [1]])
        # origin = Ti @ origin
        # dx, dy, dz = origin[0, 0], origin[1, 0], origin[2, 0]

        rf_proj = np.zeros_like(rf)
        lf_proj = np.zeros_like(lf)
        rk_proj = np.zeros_like(rk)
        lk_proj = np.zeros_like(lk)

        for i in range(self.frames + 1):
            Ts, Th, Tf, Te = self.IK_trunk(i)
            Ts_ra, Ts_re, Ts_la, Ts_le, Ts_rl, Ts_rk, Ts_ll, Ts_lk = self.IK_limbs(i, Ts, Th)

            if k == 8:
                T1, T2, T3, T4 = Ts_rk, Ts_lk, Ts_rl, Ts_ll
            elif k == 2:
                T1, T2, T3, T4 = Ts_re, Ts_le, Ts_ra, Ts_la

            # feet

            yfR = np.matrix([[0], [L2R[i]], [0], [1]])
            yfL = np.matrix([[0], [L2L[i]], [0], [1]])

            rf_proj[:, i] = (T1 @ yfR)[0:3, 0]
            lf_proj[:, i] = (T2 @ yfL)[0:3, 0]

            # knees

            ykR = np.matrix([[0], [L1R[i]], [0], [1]])
            ykL = np.matrix([[0], [L1L[i]], [0], [1]])

            rk_proj[:, i] = (T3 @ ykR)[0:3, 0]
            lk_proj[:, i] = (T4 @ ykL)[0:3, 0]

        diffRf = rf - rf_proj
        diffLf = lf - lf_proj

        diffRk = rk - rk_proj
        diffLk = lk - lk_proj

        plt.plot(np.linalg.norm(diffRf, axis=0).T)
        plt.plot(np.linalg.norm(diffLf, axis=0).T)
        plt.plot(np.linalg.norm(diffRk, axis=0).T)
        plt.plot(np.linalg.norm(diffLk, axis=0).T)

        plt.legend(["rfoot", "lfoot", "rknee", "lknee"])
        plt.show()

        thresh = 0.001

        check_lf = np.linalg.norm(diffLf, axis=0) >= thresh
        check_rf = np.linalg.norm(diffRf, axis=0) >= thresh
        check_lk = np.linalg.norm(diffLk, axis=0) >= thresh
        check_rk = np.linalg.norm(diffRk, axis=0) >= thresh

        idxl3, idxl4 = (self.thet3_ll.T[check_lf], self.thet4_ll.T[check_lf])
        idxr3, idxr4 = (self.thet3_rl.T[check_rf], self.thet4_rl.T[check_rf])

        # idxl3n, idxl4n = (self.thet3_ll.T[check_l], self.thet4_ll.T[check_l])
        # idxr3n, idxr4n = (self.thet3_rl.T[check], self.thet4_rl.T[check])

        print("Left foot: ", sum(check_lf))
        print("Right foot: ", sum(check_rf))
        #   , sum(check_lk), sum(check_rk))

        # print(np.rad2deg(np.max(idxl4)), np.rad2deg(np.min(idxl4)))
        # print(np.rad2deg(np.max(idxr4)), np.rad2deg(np.min(idxr4)))

    def show_T(self, j):
        # dx, dy, dz = self.mid_shoulder[0, j], self.mid_shoulder[1, j], self.mid_shoulder[2, j]
        Ts, Th, Tf, Te = self.IK_trunk(j)

        ##WORLD ORIGIN
        # self.ax = self.show_Ti(self.ax, np.eye(4), sign=1)

        ## SHOULDERS

        self.ax = self.show_Ti(
            self.ax,
            Ts,
            i="s",
        )

        # self.ax = self.show_Ti(
        #     self.ax,
        #     Ts,
        #     sign=-1,
        #     i="-s",
        # )

        ## HIPS
        self.ax = self.show_Ti(
            self.ax,
            Th,
            i="h",
        )

        # self.ax = self.show_Ti(
        #     self.ax,
        #     Th,
        #     sign=-1,
        #     i="-h",
        # )

        ## FACE
        # self.ax = self.show_Ti(
        #     self.ax,
        #     Tf,
        #     i="f",
        # )

        # self.ax = self.show_Ti(
        #     self.ax,
        #     Tf,
        #     sign=-1,
        #     i="-f",
        # )

        ## HEAD
        # self.ax = self.show_Ti(
        #     self.ax,
        #     Te,
        #     i="e",
        # )

        # self.ax = self.show_Ti(
        #     self.ax,
        #     Te,
        #     sign=-1,
        #     i="-e",
        # )

        Ts_ra, Ts_re, Ts_la, Ts_le, Ts_rl, Ts_rk, Ts_ll, Ts_lk = self.IK_limbs(j, Ts, Th)

        # Right elbow
        # Trs = Ts_ra
        # self.ax = self.show_Ti(self.ax, Trs, i="e", sign=1)

        # Right hand
        Tre = Ts_re
        self.ax = self.show_Ti(self.ax, Tre, i="h")

        # Left elbow
        # Tls = Ts_la
        # self.ax = self.show_Ti(self.ax, Tls, i="e", sign=1)

        # Left hand
        Tle = Ts_le
        self.ax = self.show_Ti(self.ax, Tle, i="h", sign=1)

        # Right leg
        # Trh = Ts_rl
        # self.ax = self.show_Ti(self.ax, Trh, i="l")

        # Right knee
        Trk = Ts_rk
        self.ax = self.show_Ti(self.ax, Trk, i="h")

        # Left leg
        # Tlh = Ts_ll
        # self.ax = self.show_Ti(self.ax, Tlh, i="l")

        # Left knee
        Tlk = Ts_lk
        self.ax = self.show_Ti(self.ax, Tlk, i="h")

    def IK_trunk(self, j):
        ## SHOULDERS
        Rs = self.R_zy(self.thet1s[0, j], self.thet2s[0, j])
        T0s = self.get_T(Rs, [self.mid_shoulder[0, j], self.mid_shoulder[1, j], self.mid_shoulder[2, j]])

        ## HIPS

        Rh = self.R_xzy(self.thet1h[0, j], self.thet2h[0, j], self.thet3h[0, j])
        T0h = self.get_T(Rh, [self.mid_hip[0, j], self.mid_hip[1, j], self.mid_hip[2, j]])

        # Tsh = T0s @ np.linalg.inv(T0h)
        Tsh = np.linalg.inv(T0s) @ T0h
        Th = T0s @ Tsh

        ## FACE
        Rf = self.R_xzy(self.thet1f[0, j], self.thet2f[0, j], self.thet3f[0, j])
        T0f = self.get_T(Rf, [self.mid_face[0, j], self.mid_face[1, j], self.mid_face[2, j]])

        Tsf = np.linalg.inv(T0s) @ T0f
        Tf = T0s @ Tsf

        ## HEAD
        Re = self.R_xzy(self.thet1e[0, j], self.thet2e[0, j], self.thet3e[0, j])
        T0e = self.get_T(Re, [self.mid_ear[0, j], self.mid_ear[1, j], self.mid_ear[2, j]])

        Tse = np.linalg.inv(T0s) @ T0e
        Te = T0s @ Tse

        return T0s, Th, Tf, Te

    def IK_limbs(self, j, T0_1, T0_h):

        ## RIGHT ARM
        T0_2, T0_3 = self.T_limb(
            j, 2, np.matrix([self.thet1_ra[0, j], self.thet2_ra[0, j], self.thet3_ra[0, j], self.thet4_ra[0, j]])
        )

        T1_2 = np.linalg.inv(T0_1) @ T0_2
        T02 = T0_1 @ T1_2
        T2_3 = np.linalg.inv(T02) @ T0_3
        T03 = T02 @ T2_3

        ## LEFT ARM
        T0_5, T0_6 = self.T_limb(
            j, 5, np.matrix([self.thet1_la[0, j], self.thet2_la[0, j], self.thet3_la[0, j], self.thet4_la[0, j]])
        )

        T1_5 = np.linalg.inv(T0_1) @ T0_5
        T05 = T0_1 @ T1_5
        T5_6 = np.linalg.inv(T05) @ T0_6
        T06 = T05 @ T5_6

        ## RIGHT LEG
        T0_8, T0_9 = self.T_limb(
            j, 8, np.matrix([self.thet1_rl[0, j], self.thet2_rl[0, j], self.thet3_rl[0, j], self.thet4_rl[0, j]])
        )

        Th_8 = np.linalg.inv(T0_h) @ T0_8
        T08 = T0_h @ Th_8
        T8_9 = np.linalg.inv(T08) @ T0_9
        T09 = T08 @ T8_9

        ## LEFT LEG
        T0_11, T0_12 = self.T_limb(
            j, 11, np.matrix([self.thet1_ll[0, j], self.thet2_ll[0, j], self.thet3_ll[0, j], self.thet4_ll[0, j]])
        )

        Th_11 = np.linalg.inv(T0_h) @ T0_11
        T011 = T0_h @ Th_11
        T11_12 = np.linalg.inv(T011) @ T0_12
        T012 = T011 @ T11_12

        return T02, T03, T05, T06, T08, T09, T011, T012

    def T_limb(self, j, k, T):
        t1, t2, t3, t4 = T[0, 0], T[0, 1], T[0, 2], T[0, 3]

        Rz_180 = np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

        R_03 = Rz_180 @ self.R_zxy(t1, t2, t3)
        T0_03 = self.get_T(R_03, [self.X[j, k], self.Y[j, k], self.Z[j, k]])

        # RIGHT ELBOW
        R_04 = R_03 @ self.R_x(t4)
        T0_04 = self.get_T(R_04, [self.X[j, k + 1], self.Y[j, k + 1], self.Z[j, k + 1]])

        return T0_03, T0_04

    def get_T(self, R, origin):
        T = np.zeros((4, 4))
        T[3, 3] = 1

        T[0:3, 0:3] = R
        T[0, 3] = origin[0]
        T[1, 3] = origin[1]
        T[2, 3] = origin[2]

        return T

    def show_Ti(self, ax, Ti, i=0, scale=0.1, sign=1):
        # Ri = Ti[0:3, 0:3]

        origin = np.matrix([[0], [0], [0], [1]])
        origin = Ti @ origin

        # print("dx: ", vec.T)

        dx, dy, dz = origin[0, 0], origin[1, 0], origin[2, 0]

        var = 1 * scale * sign

        xaxis = np.matrix([[var], [0], [0], [1]])
        yaxis = np.matrix([[0], [var], [0], [1]])
        zaxis = np.matrix([[0], [0], [var], [1]])

        Ri = Ti[0:3, 0:3]

        x = Ri @ xaxis[0:3, 0]
        y = Ri @ yaxis[0:3, 0]
        z = Ri @ zaxis[0:3, 0]

        # print("B4: ", xaxis * 1000)
        # print("AF: ", x * 1000)
        # print(x[1, 0])
        # print(x[2] ** 2 + x[1] ** 2)

        d = scale / 10

        ax.quiver([dx], [dy], [dz], x[0, 0], x[1, 0], x[2, 0], colors=[1, 0, 0])
        ax.text(
            x[0, 0] + dx + d * 1,
            x[1, 0] + dy + d * 1,
            x[2, 0] + dz + d * 1,
            "X" + str(i),
            size=10,
            zorder=1,
            color=[1, 0, 0],
        )

        ax.quiver([dx], [dy], [dz], y[0, 0], y[1, 0], y[2, 0], colors="green")
        ax.text(
            y[0, 0] + dx + d * 2,
            y[1, 0] + dy + d * 2,
            y[2, 0] + dz + d * 2,
            "Y" + str(i),
            size=10,
            zorder=1,
            color="green",
        )

        ax.quiver([dx], [dy], [dz], z[0, 0], z[1, 0], z[2, 0], colors=[0, 0, 1])
        ax.text(
            z[0, 0] + dx + d * 3,
            z[1, 0] + dy + d * 3,
            z[2, 0] + dz + d * 3,
            "Z" + str(i),
            size=10,
            zorder=1,
            color=[0, 0, 1],
        )

        return ax

    def plot_upper(self, j, ax, T, scale=[0.1, 0.1, 0.1], color1="black", color2="black", color3="black", alph=0.5):
        rs = np.matrix([[self.X[j, 2]], [self.Y[j, 2]], [self.Z[j, 2]]])
        ls = np.matrix([[self.X[j, 5]], [self.Y[j, 5]], [self.Z[j, 5]]])

        lx, ly, lz = scale[0], scale[1], scale[2]
        # Ri = Ti[0:3, 0:3]

        # percentage of upper trunk length above shoulders (dy)
        p1 = 0.1
        # percentage of upper trunk length above shoulders (dz)
        p2 = 0.5

        dx = 0
        dy = ly * p1
        dz = lz * p2

        dy_ = -ly * (1 - p1)
        dz_ = -lz * (1 - p2)

        R = T[0:3, 0:3]

        # print(rs)
        # print(R)

        a = rs + (R @ np.matrix([[0], [dy], [dz]]))
        b = ls + (R @ np.matrix([[0], [dy], [dz]]))
        c = rs + (R @ np.matrix([[0], [dy], [dz_]]))
        d = ls + (R @ np.matrix([[0], [dy], [dz_]]))
        e = rs + (R @ np.matrix([[0], [dy_], [dz]]))
        f = ls + (R @ np.matrix([[0], [dy_], [dz]]))
        g = rs + (R @ np.matrix([[0], [dy_], [dz_]]))
        h = ls + (R @ np.matrix([[0], [dy_], [dz_]]))

        # X lines
        ax.plot([a[0, 0], b[0, 0]], [a[1, 0], b[1, 0]], [a[2, 0], b[2, 0]], color=color1, alpha=alph)
        ax.plot([e[0, 0], f[0, 0]], [e[1, 0], f[1, 0]], [e[2, 0], f[2, 0]], color=color1, alpha=alph)
        ax.plot([c[0, 0], d[0, 0]], [c[1, 0], d[1, 0]], [c[2, 0], d[2, 0]], color=color1, alpha=alph)
        ax.plot([g[0, 0], h[0, 0]], [g[1, 0], h[1, 0]], [g[2, 0], h[2, 0]], color=color1, alpha=alph)

        # Z lines
        ax.plot([a[0, 0], c[0, 0]], [a[1, 0], c[1, 0]], [a[2, 0], c[2, 0]], color=color2, alpha=alph)
        ax.plot([e[0, 0], g[0, 0]], [e[1, 0], g[1, 0]], [e[2, 0], g[2, 0]], color=color2, alpha=alph)
        ax.plot([b[0, 0], d[0, 0]], [b[1, 0], d[1, 0]], [b[2, 0], d[2, 0]], color=color2, alpha=alph)
        ax.plot([f[0, 0], h[0, 0]], [f[1, 0], h[1, 0]], [f[2, 0], h[2, 0]], color=color2, alpha=alph)

        # # Y lines
        ax.plot([a[0, 0], e[0, 0]], [a[1, 0], e[1, 0]], [a[2, 0], e[2, 0]], color=color3, alpha=alph)
        ax.plot([c[0, 0], g[0, 0]], [c[1, 0], g[1, 0]], [c[2, 0], g[2, 0]], color=color3, alpha=alph)
        ax.plot([b[0, 0], f[0, 0]], [b[1, 0], f[1, 0]], [b[2, 0], f[2, 0]], color=color3, alpha=alph)
        ax.plot([d[0, 0], h[0, 0]], [d[1, 0], h[1, 0]], [d[2, 0], h[2, 0]], color=color3, alpha=alph)

        return ax

    def plot_lower(self, j, ax, T, scale=[0.1, 0.1, 0.1], color1="black", color2="black", color3="black", alph=0.5):
        rs = np.matrix([[self.X[j, 8]], [self.Y[j, 8]], [self.Z[j, 8]]])
        ls = np.matrix([[self.X[j, 11]], [self.Y[j, 11]], [self.Z[j, 11]]])

        lx, ly, lz = scale[0], scale[1], scale[2]
        # Ri = Ti[0:3, 0:3]

        # percentage of lower trunk length above shoulders (dy)
        p1 = 0.65
        # percentage of lower trunk length above shoulders (dz)
        p2 = 0.5

        dx = 0.02
        dy = ly * p1
        dz = lz * p2

        dy_ = -ly * (1 - p1)
        dz_ = -lz * (1 - p2)

        R = T[0:3, 0:3]

        # print(rs)
        # print(R)

        a = rs + (R @ np.matrix([[-dx], [dy], [dz]]))
        b = ls + (R @ np.matrix([[dx], [dy], [dz]]))
        c = rs + (R @ np.matrix([[-dx], [dy], [dz_]]))
        d = ls + (R @ np.matrix([[dx], [dy], [dz_]]))
        e = rs + (R @ np.matrix([[-dx], [dy_], [dz]]))
        f = ls + (R @ np.matrix([[dx], [dy_], [dz]]))
        g = rs + (R @ np.matrix([[-dx], [dy_], [dz_]]))
        h = ls + (R @ np.matrix([[dx], [dy_], [dz_]]))

        dpx = lx * 0.3
        dpy = ly * 0.45

        A = e + (R @ np.matrix([[0], [dpy], [0]]))
        B = f + (R @ np.matrix([[0], [dpy], [0]]))
        C = e + (R @ np.matrix([[dpx], [0], [0]]))
        D = f + (R @ np.matrix([[-dpx], [0], [0]]))
        E = g + (R @ np.matrix([[0], [dpy], [0]]))
        F = h + (R @ np.matrix([[0], [dpy], [0]]))
        G = g + (R @ np.matrix([[dpx], [0], [0]]))
        H = h + (R @ np.matrix([[-dpx], [0], [0]]))

        # X lines
        ax.plot([a[0, 0], b[0, 0]], [a[1, 0], b[1, 0]], [a[2, 0], b[2, 0]], color=color1, alpha=alph)
        ax.plot([C[0, 0], D[0, 0]], [C[1, 0], D[1, 0]], [C[2, 0], D[2, 0]], color=color1, alpha=alph)
        ax.plot([c[0, 0], d[0, 0]], [c[1, 0], d[1, 0]], [c[2, 0], d[2, 0]], color=color1, alpha=alph)
        ax.plot([G[0, 0], H[0, 0]], [G[1, 0], H[1, 0]], [G[2, 0], H[2, 0]], color=color1, alpha=alph)

        # Y lines
        ax.plot([a[0, 0], A[0, 0]], [a[1, 0], A[1, 0]], [a[2, 0], A[2, 0]], color=color2, alpha=alph)
        ax.plot([c[0, 0], E[0, 0]], [c[1, 0], E[1, 0]], [c[2, 0], E[2, 0]], color=color2, alpha=alph)
        ax.plot([b[0, 0], B[0, 0]], [b[1, 0], B[1, 0]], [b[2, 0], B[2, 0]], color=color2, alpha=alph)
        ax.plot([d[0, 0], F[0, 0]], [d[1, 0], F[1, 0]], [d[2, 0], F[2, 0]], color=color2, alpha=alph)

        # Z lines
        ax.plot([a[0, 0], c[0, 0]], [a[1, 0], c[1, 0]], [a[2, 0], c[2, 0]], color=color3, alpha=alph)
        ax.plot([b[0, 0], d[0, 0]], [b[1, 0], d[1, 0]], [b[2, 0], d[2, 0]], color=color3, alpha=alph)

        # Vertical lines
        ax.plot([C[0, 0], A[0, 0]], [C[1, 0], A[1, 0]], [C[2, 0], A[2, 0]], color=color3, alpha=alph)
        ax.plot([G[0, 0], E[0, 0]], [G[1, 0], E[1, 0]], [G[2, 0], E[2, 0]], color=color3, alpha=alph)
        ax.plot([D[0, 0], B[0, 0]], [D[1, 0], B[1, 0]], [D[2, 0], B[2, 0]], color=color3, alpha=alph)
        ax.plot([H[0, 0], F[0, 0]], [H[1, 0], F[1, 0]], [H[2, 0], F[2, 0]], color=color3, alpha=alph)

        # More Z lines
        ax.plot([E[0, 0], A[0, 0]], [E[1, 0], A[1, 0]], [E[2, 0], E[2, 0]], color=color3, alpha=alph)
        ax.plot([G[0, 0], C[0, 0]], [G[1, 0], C[1, 0]], [G[2, 0], C[2, 0]], color=color3, alpha=alph)
        ax.plot([F[0, 0], B[0, 0]], [F[1, 0], B[1, 0]], [F[2, 0], B[2, 0]], color=color3, alpha=alph)
        ax.plot([H[0, 0], D[0, 0]], [H[1, 0], D[1, 0]], [H[2, 0], D[2, 0]], color=color3, alpha=alph)

        return ax

    def plot_head(self, j, ax, T, radius=0.07, color="black", alph=0.5):
        theta = np.linspace(0, 2 * np.pi, 100)
        x = np.sin(theta) * radius
        z = np.cos(theta) * radius
        y = np.zeros_like(theta)

        # Rotation matrix (rotate around Y-axis)
        R = T[0:3, 0:3]

        # Upper Limb
        c_1 = np.dot(R, np.array([x, y, z])) + np.matrix(T[0:3, 3]).T
        c1 = np.zeros_like(c_1)
        c2 = np.zeros_like(c_1)

        c1 += c_1 + (R @ np.matrix([[0], [0.05], [0]]))
        ax.plot(c1[0], c1[1], c1[2], color=color, alpha=alph)
        c2 += c_1 + (R @ np.matrix([[0], [-0.02], [0]]))
        ax.plot(c2[0], c2[1], c2[2], color=color, alpha=alph)

        for i in [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:
            ax.plot([c1[0, i], c2[0, i]], [c1[1, i], c2[1, i]], [c1[2, i], c2[2, i]], color=color, alpha=alph)

        return ax

    def plot_limbs(self, j, k, ax, T1, T2, radius=0.025, color1="black", color2="black", alph=0.5):
        r1 = np.matrix([self.X[j, k], self.Y[j, k], self.Z[j, k]])
        r2 = np.matrix([self.X[j, k + 1], self.Y[j, k + 1], self.Z[j, k + 1]])
        r3 = np.matrix([self.X[j, k + 2], self.Y[j, k + 2], self.Z[j, k + 2]])

        theta = np.linspace(0, 2 * np.pi, 100)
        x = np.sin(theta) * radius
        z = np.cos(theta) * radius
        y = np.zeros_like(theta)

        # Rotation matrix (rotate around Y-axis)
        R1 = T1[0:3, 0:3]
        R2 = T2[0:3, 0:3]

        # Upper Limb
        c_1 = np.dot(R1, np.array([x, y, z]))
        c1 = np.zeros_like(c_1)
        c2 = np.zeros_like(c_1)

        c1 = c_1 + r1.T
        ax.plot(c1[0], c1[1], c1[2], color=color2, alpha=alph)
        c2 = c_1 + r2.T
        ax.plot(c2[0], c2[1], c2[2], color=color2, alpha=alph)

        # Lower Limb
        c_2 = np.dot(R2, np.array([x, y, z]))
        c3 = np.zeros_like(c_2)
        c4 = np.zeros_like(c_2)
        c3 = c_2 + r2.T
        ax.plot(c3[0], c3[1], c3[2], color=color2, alpha=alph)
        c4 = c_2 + r3.T
        ax.plot(c4[0], c4[1], c4[2], color=color2, alpha=alph)

        for i in [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:
            ax.plot([c1[0, i], c2[0, i]], [c1[1, i], c2[1, i]], [c1[2, i], c2[2, i]], color=color1, alpha=alph)
            ax.plot([c3[0, i], c4[0, i]], [c3[1, i], c4[1, i]], [c3[2, i], c4[2, i]], color=color2, alpha=alph)

        return ax

    def model_overlay(self, j):
        Ts, Th, Tf, Te = self.IK_trunk(j)
        Ts_ra, Ts_re, Ts_la, Ts_le, Ts_rl, Ts_rk, Ts_ll, Ts_lk = self.IK_limbs(j, Ts, Th)

        self.ax = self.plot_upper(j=j, ax=self.ax, T=Ts, scale=[0.1, 0.1, 0.1])
        self.ax = self.plot_lower(j=j, ax=self.ax, T=Th, scale=[0.1, 0.1, 0.1])
        self.ax = self.plot_head(j=j, ax=self.ax, T=Te)
        self.ax = self.plot_limbs(j=j, k=2, ax=self.ax, T1=Ts_ra, T2=Ts_re, radius=0.025)
        self.ax = self.plot_limbs(j=j, k=5, ax=self.ax, T1=Ts_la, T2=Ts_le, radius=0.025)
        self.ax = self.plot_limbs(j=j, k=8, ax=self.ax, T1=Ts_rl, T2=Ts_rk, radius=0.035)
        self.ax = self.plot_limbs(j=j, k=11, ax=self.ax, T1=Ts_ll, T2=Ts_lk, radius=0.035)

    def IK_init(self):

        ## SHOULDERS
        self.shoudler_ang()

        ## HIPS
        self.hip_ang()

        ## FACE
        self.face_ang()

        ## HEAD
        self.head_ang()

        ##ARMS
        self.thet1_ra, self.thet2_ra, self.thet3_ra, self.thet4_ra = self.arm_ang(k=2)
        self.thet1_la, self.thet2_la, self.thet3_la, self.thet4_la = self.arm_ang(k=5)
        ##LEGS
        self.thet1_rl, self.thet2_rl, self.thet3_rl, self.thet4_rl = self.leg_ang(k=8)
        self.thet1_ll, self.thet2_ll, self.thet3_ll, self.thet4_ll = self.leg_ang(k=11)

    def ID_init(self):
        T_ra = np.array([self.thet1_ra, self.thet2_ra, self.thet3_ra, self.thet4_ra])[:, 0, :]
        T_la = np.array([self.thet1_la, self.thet2_la, self.thet3_la, self.thet4_la])[:, 0, :]

        T_rl = np.array([self.thet1_rl, self.thet2_rl, self.thet3_rl, self.thet4_rl])[:, 0, :]
        T_ll = np.array([self.thet1_ll, self.thet2_ll, self.thet3_ll, self.thet4_ll])[:, 0, :]

    def inv_dynamics_upper(self, L, i1):
        dt = 1 / 30
        T = np.array([self.thet1s, self.thet2s])[:, 0, :]
        TD = np.gradient(T, dt, axis=1)
        TDD = np.gradient(TD, dt, axis=1)

        # print(np.shape(TD))
        ###### FORWARD ITERATION
        w0 = np.zeros([3, 1])
        w0_d = np.zeros([3, 1])
        v0_d = np.zeros([3, 1])
        v0_d[2] = -9.81

        n = self.frames + 1

        Tq = np.ones((3, n))

        W = np.ones((3, n))
        W_d = np.ones((3, n))
        Vls_d = np.ones((3, n))
        Vrs_d = np.ones((3, n))

        # plt.plot(np.rad2deg(T[0, :]))
        # plt.plot(np.rad2deg(T[1, :]))
        # plt.legend(["t1", "t2"])
        # plt.show()

        for i in range(n):
            t1, t2 = T[0, i], T[1, i]
            t1d, t2d = TD[0, i], TD[1, i]
            t1dd, t2dd = TDD[0, i], TDD[1, i]

            # # initializing paramteres from refrence to first frame

            ## ANGULAR
            # angular velocity

            Ry1 = self.R_y(t1)
            Rz2 = self.R_z(t2)

            R01 = Ry1 @ Rz2

            p_ls = np.matrix([[L], [0], [0]])
            p_rs = np.matrix([[-L], [0], [0]])

            w1 = Ry1.T @ w0 + np.matrix([[0], [t1d], [0]])
            w2 = Rz2.T @ w1 + np.matrix([[0], [0], [t2d]])

            # angular accelearation
            w1_d = (
                (Ry1.T @ w0_d)
                + np.cross(Ry1.T @ w0, np.matrix([[0], [t1d], [0]]), axis=0)
                + np.matrix([[0], [t1dd], [0]])
            )
            w2_d = (
                (Rz2.T @ w1_d)
                + np.cross(Rz2.T @ w1, np.matrix([[0], [0], [t2d]]), axis=0)
                + np.matrix([[0], [0], [t2dd]])
            )

            ## LINEAR
            # linear accelearation
            vls_d = np.cross(w2_d, p_ls / 2, axis=0) + np.cross(w2, np.cross(w2, p_ls / 2, axis=0), axis=0) + v0_d
            vrs_d = np.cross(w2_d, p_rs / 2, axis=0) + np.cross(w2, np.cross(w2, p_rs / 2, axis=0), axis=0) + v0_d

            ##MOMENTS
            I = np.diag(np.ravel(i1))
            N = (I @ w2_d) + np.cross(w2, (I @ w2), axis=0)

            Tq[:, [i]] = (N)[:, 0]

            W[:, [i]] = (w2)[:, 0]
            W_d[:, [i]] = (w2_d)[:, 0]

            Vls_d[:, [i]] = vls_d
            Vrs_d[:, [i]] = vrs_d

        return Tq, W, W_d, Vls_d, Vrs_d

    def inv_dynamics_lower(self, L, i1, to_upper=False):
        dt = 1 / 30
        T = np.array([self.thet1h, self.thet2h, self.thet3h])[:, 0, :]
        TD = np.gradient(T, dt, axis=1)
        TDD = np.gradient(TD, dt, axis=1)

        # print(np.shape(TD))
        ###### FORWARD ITERATION
        w0 = np.zeros([3, 1])
        w0_d = np.zeros([3, 1])
        v0_d = np.zeros([3, 1])
        v0_d[2] = -9.81

        n = self.frames + 1

        Tq = np.ones((3, n))

        W = np.ones((3, n))
        W_d = np.ones((3, n))
        Vlh_d = np.ones((3, n))
        Vrh_d = np.ones((3, n))

        # plt.plot(np.rad2deg(T[0, :]))
        # plt.plot(np.rad2deg(T[1, :]))
        # plt.plot(np.rad2deg(T[2, :]))
        # plt.legend(["t1", "t2", "t3"])
        # plt.show()

        for i in range(n):
            t1, t2, t3 = T[0, i], T[1, i], T[2, i]
            t1d, t2d, t3d = TD[0, i], TD[1, i], TD[2, i]
            t1dd, t2dd, t3dd = TDD[0, i], TDD[1, i], TDD[2, i]

            # # initializing paramteres from refrence to first frame

            ## ANGULAR
            # angular velocity

            Rx1 = self.R_x(t1)
            Rz2 = self.R_z(t2)
            Ry3 = self.R_y(t3)

            R01 = Rx1 @ Rz2 @ Ry3

            p_lh = np.matrix([[L], [0], [0]])
            p_rh = np.matrix([[-L], [0], [0]])

            w1 = Rx1.T @ w0 + np.matrix([[t1d], [0], [0]])
            w2 = Rz2.T @ w1 + np.matrix([[0], [0], [t2d]])
            w3 = Ry3.T @ w2 + np.matrix([[0], [t3d], [0]])

            # angular accelearation
            w1_d = (
                (Rx1.T @ w0_d)
                + np.cross(Rx1.T @ w0, np.matrix([[t1d], [0], [0]]), axis=0)
                + np.matrix([[t1dd], [0], [0]])
            )
            w2_d = (
                (Rz2.T @ w1_d)
                + np.cross(Rz2.T @ w1, np.matrix([[0], [0], [t2d]]), axis=0)
                + np.matrix([[0], [0], [t2dd]])
            )
            w3_d = (
                (Ry3.T @ w2_d)
                + np.cross(Ry3.T @ w2, np.matrix([[0], [t3d], [0]]), axis=0)
                + np.matrix([[0], [t3dd], [0]])
            )

            ## LINEAR
            # linear accelearation
            vlh_d = np.cross(w3_d, p_lh / 2, axis=0) + np.cross(w3, np.cross(w3, p_lh / 2, axis=0), axis=0) + v0_d
            vrh_d = np.cross(w3_d, p_rh / 2, axis=0) + np.cross(w3, np.cross(w3, p_rh / 2, axis=0), axis=0) + v0_d

            ##MOMENTS
            I = np.diag(np.ravel(i1))
            N = (I @ w3_d) + np.cross(w3, (I @ w3), axis=0)

            Tq[:, [i]] = (N)[:, 0]

            W[:, [i]] = (w2)[:, 0]
            W_d[:, [i]] = (w2_d)[:, 0]

            # torque from shoulder flextion/extension
            vlh_d = R01 @ vlh_d
            # forces transformed to base frame
            vrh_d = R01 @ vrh_d

            if to_upper:
                T0s, T0h, T0f, T0e = self.IK_trunk(i)

                Vlh_d[:, [i]] = T0s[0:3, 0:3].T @ vlh_d
                Vrh_d[:, [i]] = T0s[0:3, 0:3].T @ vrh_d
            else:
                Vlh_d[:, [i]] = vlh_d
                Vrh_d[:, [i]] = vrh_d

        return Tq, W, W_d, Vlh_d, Vrh_d

    def inv_dynamics(self, T, L, m, I, W0=None, W0_d=None, V0_d=None, to_upper=False, to_lower=False):

        # if k == 2 or k == 5:
        #     thet1, thet2, thet3, thet4 = self.arm_ang(k)
        # elif k == 8 or k == 11:
        #     thet1, thet2, thet3, thet4 = self.leg_ang(k)

        dt = 1 / 30

        # T = np.array([thet1, thet2, thet3, thet4])[:, 0, :]

        # TD = np.gradient(T, dt, axis=1)
        # TDD = np.gradient(TD, dt, axis=1)

        win = 5
        order = 1

        T = scipy.signal.savgol_filter(T, win, order, axis=1)
        TD = scipy.signal.savgol_filter(T, win, order, axis=1, deriv=1, delta=dt)
        TDD = scipy.signal.savgol_filter(T, win, order, axis=1, deriv=2, delta=dt)

        n = self.frames + 1

        # print(np.shape(TD))
        ###### FORWARD ITERATION

        # t1, t2, t3, t4 = T[:, 0], T[:, 1], T[:, 2], T[:, 3]
        # t1d, t2d, t3d, t4d = TD[:, 0], TD[:, 1], TD[:, 2], TD[:, 3]
        # t1dd, t2dd, t3dd, t4dd = TDD[:, 0], TDD[:, 1], TDD[:, 2], TDD[:, 3]

        ###### FORWARD ITERATION
        if W0 is None:
            W0 = np.zeros((3, n))
        if W0_d is None:
            W0_d = np.zeros((3, n))
        if V0_d is None:
            V0_d = np.zeros((3, n))
            V0_d[2, :] = -9.81

        L1, L2 = L[0], L[1]
        m1, m2 = m[0], m[1]
        i1, i2 = I[0, :], I[1, :]

        Fr = np.ones((3, n))
        Tq = np.ones((3, n))

        extra1 = np.ones((3, n))
        extra2 = np.ones((3, n))

        for i in range(n):
            t1, t2, t3, t4 = T[0, i], T[1, i], T[2, i], T[3, i]
            t1d, t2d, t3d, t4d = TD[0, i], TD[1, i], TD[2, i], TD[3, i]
            t1dd, t2dd, t3dd, t4dd = TDD[0, i], TDD[1, i], TDD[2, i], TDD[3, i]

            w0 = W0[:, i].reshape(3, 1)
            w0_d = W0_d[:, i].reshape(3, 1)
            v0_d = V0_d[:, i].reshape(3, 1)

            # # initializing paramteres from refrence to first frame

            ## ANGULAR
            # angular velocity
            Rz_180 = np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

            Rz1 = Rz_180 @ self.R_z(t1)
            Rx2 = self.R_x(t2)
            Ry3 = self.R_y(t3)
            Rx4 = self.R_x(t4)

            R01 = Rz_180 @ self.R_zxy(t1, t2, t3)
            R12 = Rx4

            p1 = np.matrix([[0], [L1], [0]])
            p2 = np.matrix([[0], [L2], [0]])

            wz = Rz1.T @ w0 + np.matrix([[0], [0], [t1d]])
            wx = Rx2.T @ wz + np.matrix([[t2d], [0], [0]])
            w1 = Ry3.T @ wx + np.matrix([[0], [t3d], [0]])
            w2 = Rx4.T @ w1 + np.matrix([[t4d], [0], [0]])

            # angular accelearation
            wz_d = (
                (Rz1.T @ w0_d)
                + np.cross(Rz1.T @ w0, np.matrix([[0], [0], [t1d]]), axis=0)
                + np.matrix([[0], [0], [t1dd]])
            )
            wx_d = (
                (Rx2.T @ wz_d)
                + np.cross(Rx2.T @ wz, np.matrix([[t2d], [0], [0]]), axis=0)
                + np.matrix([[t2dd], [0], [0]])
            )
            w1_d = (
                (Ry3.T @ wx_d)
                + np.cross(Ry3.T @ wx, np.matrix([[0], [t3d], [0]]), axis=0)
                + np.matrix([[0], [t3dd], [0]])
            )
            w2_d = (
                (Rx4.T @ w1_d)
                + np.cross(Rx4.T @ w1, np.matrix([[t4d], [0], [0]]), axis=0)
                + np.matrix([[t4dd], [0], [0]])
            )
            ## LINEAR

            # linear accelearation
            v1_d = R01.T @ v0_d
            v2_d = R12.T @ (np.cross(w1_d, p1, axis=0) + np.cross(w1, np.cross(w1, p1, axis=0), axis=0) + v1_d)

            # linear acceleration relative to COM
            vc1_d = np.cross(w1_d, p1 / 2, axis=0) + np.cross(w1, np.cross(w1, p1 / 2, axis=0), axis=0) + v1_d
            vc2_d = np.cross(w2_d, p2 / 2, axis=0) + np.cross(w2, np.cross(w2, p2 / 2, axis=0), axis=0) + v2_d

            ## FORCES
            # froces from shoulder frames
            F1 = m1 * vc1_d
            # froces from ekbow frame
            F2 = m2 * vc2_d

            ##MOMENTS
            I1 = np.diag(np.ravel(i1))
            N1 = (I1 @ w1_d) + np.cross(w1, (I1 @ w1), axis=0)
            I2 = np.diag(np.ravel(i2))
            N2 = (I2 @ w2_d) + np.cross(w2, (I2 @ w2), axis=0)

            ###### BACKWARDS ITERATION
            # ACCUMULATED FORCES at joints
            f2 = F2
            f1 = F1 + (R12 @ f2)

            # ACCUMULATED TORQUE at joints
            n2 = N2 + np.cross(p2 / 2, F2, axis=0)
            n1 = N1 + (R12 @ n2) + np.cross(p1 / 2, F1, axis=0) + np.cross(p1, (R12 @ f2), axis=0)

            # torque from shoulder flextion/extension
            tq = (R01 @ n1)[:, 0]
            # forces transformed to base frame
            fr = (R01 @ f1)[:, 0]

            extra1[:, [i]] = (R01 @ w1)[:, 0]
            extra2[:, [i]] = (R01 @ w2)[:, 0]

            if to_upper:
                T0s, T0h, Tf, Te = self.IK_trunk(i)

                tq = T0s[0:3, 0:3].T @ tq
                fr = T0s[0:3, 0:3].T @ fr

            if to_lower:
                T0s, T0h, Tf, Te = self.IK_trunk(i)

                tq = T0h[0:3, 0:3].T @ tq
                fr = T0h[0:3, 0:3].T @ fr

            Tq[:, [i]] = tq
            Fr[:, [i]] = fr

        return Tq, Fr, extra1, extra2


# fi = r"C:\\Users\\franc\Documents\\GitHub\\PANDA-Gym-Data-Proceeing\\Calibration\\3D_vid_2_6.csv"
# fi = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Cameras\sim_cam_2_3_vid_3.csv"


# tt = processpose(fi)

# tt.IK_init_static()

# tt.IK_init()
# tt.ID_init()

# tt.plotskel_loop(start=0 * 30)
# tt.check_IK_legs(k=2)
# tt.check_IK_legs(k=8)
# tt.check_IK_all()
