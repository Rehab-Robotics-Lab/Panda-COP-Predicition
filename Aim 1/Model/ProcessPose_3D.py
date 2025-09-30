import numpy as np
import math
import scipy
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

        # creating global variables

        X_init, Y_init, Z_init = np.mean(X[0:5, 1]), np.mean(Y[0:5, 1]), np.mean(Z[0:5, 1])

        self.X = X - X_init
        self.Y = Y - Y_init
        self.Z = Z
        self.idx = idx

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

        # s1 = "t3= " + str(int(np.rad2deg(self.thet3_ra[0, j])))
        s2 = "zmin= " + str(np.min(self.Z[j, :]) * 1000)

        # self.ax.text(0.3, -0.3, 0, "%s" % (s1), size=20, zorder=1, color="k")
        self.ax.text(0.3, -0.4, 0, "%s" % (s2), size=20, zorder=1, color="k")

        # plt.text(0, 0, s)
        self.ax.set_zlim3d(0, 0.6)
        self.ax.set_ylim3d(-0.4, 0.2)
        self.ax.set_xlim3d(-0.3, 0.3)
        plt.grid()

    def plotskel_loop(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")

        ani = animation.FuncAnimation(self.fig, self.plotskel, frames=self.frames, interval=2)
        plt.show()

    def arms(self, k):
        ls = np.matrix([self.X[:, k], self.Y[:, k], self.Z[:, k]])
        le = np.matrix([self.X[:, k + 1], self.Y[:, k + 1], self.Z[:, k + 1]])
        lh = np.matrix([self.X[:, k + 2], self.Y[:, k + 2], self.Z[:, k + 2]])

        L1 = np.linalg.norm(le - ls, axis=0)
        L2 = np.linalg.norm(le - lh, axis=0)

        print(k, np.mean(L2) * 100)

        s1 = np.divide(le[0, :] - ls[0, :], L1)
        c1 = np.divide(ls[1, :] - le[1, :], L1)

        thet1 = np.arctan2(s1, c1)

        s2 = np.divide(le[2, :] - ls[2, :], L1)
        c2 = np.divide((ls[1, :] - le[1, :]), np.multiply(c1, L1))

        thet2 = np.arctan2(s2, c2)

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
        # # s4 = np.clip(s4, -1, 1)
        c4 = np.divide(dot_product, np.multiply(magnitude_u, magnitude_v))
        # # c4 = np.clip(c4, -1, 1)

        # fining c4 with law of cosines
        # c4 = np.divide(
        #     (np.power(magnitude_u, 2) + np.power(magnitude_v, 2) - np.power(magnitude_q, 2)),
        #     (2 * np.multiply(magnitude_u, magnitude_v)),
        # )
        # s4 = np.sqrt((np.ones_like(c4) - np.power(c4, 2))).T
        # thet4 = np.pi - np.arccos(c4)

        thet4 = np.arctan2(s4, c4)

        ##THETA 3

        X = np.divide(lh[0, :] - le[0, :], L2)
        Y = np.divide(lh[1, :] - le[1, :], L2)
        Z = np.divide(lh[2, :] - le[2, :], L2)

        s3 = np.divide(-(np.multiply(X, c1) + np.multiply(Y, s1)), s4)
        # c3 = np.divide((np.multiply(c2, c4) + np.multiply(Y, c1) - np.multiply(X, s1)), np.multiply(s2, s4))
        c3 = np.sqrt((np.ones_like(s3) - np.power(s3, 2)).T).T

        thet3 = np.arctan2(s3, c3)

        # print([np.min(c3), np.max(c3)], [np.min(s3), np.max(s3)])
        # print([np.min(c4), np.max(c4)], [np.min(s4), np.max(s4)])
        return thet1, thet2, thet3, thet4

    def legs(self, k):
        ls = np.matrix([self.X[:, k], self.Y[:, k], self.Z[:, k]])
        le = np.matrix([self.X[:, k + 1], self.Y[:, k + 1], self.Z[:, k + 1]])
        lh = np.matrix([self.X[:, k + 2], self.Y[:, k + 2], self.Z[:, k + 2]])

        L1 = np.linalg.norm(le - ls, axis=0)
        L2 = np.linalg.norm(le - lh, axis=0)

        s1 = np.divide(le[0, :] - ls[0, :], L1)
        c1 = np.divide(ls[1, :] - le[1, :], L1)

        thet1 = np.arctan2(s1, c1)

        s2 = np.divide(le[2, :] - ls[2, :], L1)
        c2 = np.divide((ls[1, :] - le[1, :]), np.multiply(c1, L1))

        thet2 = np.arctan2(s2, c2)

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
        # # s4 = np.clip(s4, -1, 1)
        c4 = np.divide(dot_product, np.multiply(magnitude_u, magnitude_v))
        # # c4 = np.clip(c4, -1, 1)

        thet4 = np.arctan2(s4, c4)

        ##THETA 3

        X = np.divide(lh[0, :] - le[0, :], L2)
        Y = np.divide(lh[1, :] - le[1, :], L2)
        Z = np.divide(lh[2, :] - le[2, :], L2)

        s3 = np.divide(-(np.multiply(X, c1) + np.multiply(Y, s1)), s4)
        c3 = np.divide((np.multiply(c2, c4) + np.multiply(Y, c1) - np.multiply(X, s1)), np.multiply(s2, s4))
        c3 = -np.sqrt((np.ones_like(s3) - np.power(s3, 2)).T).T

        thet3 = np.arctan2(s3, c3)

        # print([np.min(c3), np.max(c3)], [np.min(s3), np.max(s3)])
        # print([np.min(c4), np.max(c4)], [np.min(s4), np.max(s4)])
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
        self.thet1f = thet1

        c2 = np.divide((mid[1, :] - nose[1, :]), np.cos(thet1))
        s2 = nose[0, :] - mid[0, :]

        thet2 = np.arctan2(s2, c2)
        self.thet2f = thet2

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
        self.thet1e = thet1

        c2 = np.divide((mid[1, :] - neck[1, :]), np.cos(thet1))
        s2 = neck[0, :] - mid[0, :]

        thet2 = np.arctan2(s2, c2)
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

        c2 = np.divide((neck[1, :] - mid[1, :]), c1)
        s2 = mid[0, :] - neck[0, :]

        thet2 = np.arctan2(s2, c2)
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
        self.ax = self.show_Ti(
            self.ax,
            Te,
            i="e",
        )

        # self.ax = self.show_Ti(
        #     self.ax,
        #     Te,
        #     sign=-1,
        #     i="-e",
        # )

        Ts_ra, Ts_re, Ts_la, Ts_le, Ts_rl, Ts_rk, Ts_ll, Ts_lk = self.IK_limbs(j, Ts, Th)

        # Right elbow
        Trs = Ts_ra
        self.ax = self.show_Ti(self.ax, Trs, i="e", sign=1)

        # Right hand
        Tre = Ts_re
        self.ax = self.show_Ti(self.ax, Tre, i="h")

        # Left elbow
        Tls = Ts_la
        self.ax = self.show_Ti(self.ax, Tls, i="e", sign=1)

        # Left hand
        Tle = Ts_le
        self.ax = self.show_Ti(self.ax, Tle, i="h", sign=1)

        # Right leg
        Trh = Ts_rl
        self.ax = self.show_Ti(self.ax, Trh, i="l")

        # Right knee
        Trk = Ts_rk
        self.ax = self.show_Ti(self.ax, Trk, i="h")

        # Left leg
        Tlh = Ts_ll
        self.ax = self.show_Ti(self.ax, Tlh, i="l")

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

    def T_limb(self, j, k, T, R=np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
        t1, t2, t3, t4 = T[0, 0], T[0, 1], T[0, 2], T[0, 3]

        Rz_180 = np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

        R_03 = R @ Rz_180 @ self.R_zxy(t1, t2, t3)
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
        self.thet1_ra, self.thet2_ra, self.thet3_ra, self.thet4_ra = self.arms(k=2)
        self.thet1_la, self.thet2_la, self.thet3_la, self.thet4_la = self.arms(k=5)
        ##LEGS
        self.thet1_rl, self.thet2_rl, self.thet3_rl, self.thet4_rl = self.legs(k=8)
        self.thet1_ll, self.thet2_ll, self.thet3_ll, self.thet4_ll = self.legs(k=11)


fi = r"C:\\Users\\franc\Documents\\GitHub\\PANDA-Gym-Data-Proceeing\\Calibration\\3D_vid_2_6.csv"

tt = processpose(fi)

tt.IK_init()
# p1 = tt.plo1
# p2 = tt.plo2

# plt.plot((p1.T))
# plt.plot((p2.T))

# plt.plot(np.cos(tt4.T))
# plt.show()

tt.plotskel_loop()
