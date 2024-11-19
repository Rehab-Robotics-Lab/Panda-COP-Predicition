import numpy as np
import math
import scipy
import pandas as pd
from numpy import radians as radians
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


class processPose:
    def __init__(self, file):

        df = pd.read_pickle(file)

        df["x"] = df["x"].astype(float)
        df["y"] = df["y"].astype(float)

        frames = int(np.max(df.frame))
        self.frames = frames

        empty = np.zeros((13, frames + 1))

        Xparts = np.zeros((13, frames + 1))
        Yparts = np.zeros((13, frames + 1))

        conf = 0.55

        # # looking one frame at a time
        # for f in range(frames):

        #     # finidng rows with relevant frame number
        #     rows = df.frame == f

        #     # only looking at table with relevant frame number
        #     table = df[rows]

        #     # looking one part at a time
        #     for p in range(1, 14):
        #         # boolean for if body part exists in frame and if it's confidence is beyind threshold
        #         part = any(table.part_idx == p) & all(table.c > conf)

        #         # if part is found with adequate confindence
        #         if part == 1:
        #             loc = table.part_idx == p

        #             Xparts[p - 1, f] = table.x[loc]
        #             Yparts[p - 1, f] = table.y[loc]

        ##finds/arrange X and Y coords into (13, frames) assuming no points are missing or have multiples
        for p in range(1, 14):
            # print(p)
            Xparts[p - 1, :] = df.x[df.part_idx == p]
            Yparts[p - 1, :] = df.y[df.part_idx == p]

        print("done")

        # creating global variables
        self.Xparts = Xparts
        self.Yparts = Yparts
        self.frames = frames

        # 15 window median filter
        self.Xfilt = scipy.ndimage.median_filter(Xparts, size=[1, 15])
        self.Yfilt = scipy.ndimage.median_filter(Yparts, size=[1, 15])

        # zeroing coords. to neck position
        Xzero = self.Xfilt - np.mean(self.Xfilt[0, 0:15])
        Yzero = self.Yfilt - np.mean(self.Yfilt[0, 0:15])
        # global variable for zeroed coords.
        self.Xzero = Xzero
        self.Yzero = Yzero

        # midpoint between both hips
        midpointX = (Xzero[7, :] + Xzero[10, :]) / 2
        midpointY = (Yzero[7, :] + Yzero[10, :]) / 2

        # trunk length (from neck to hip midpoint)
        tlengthX = Xzero[0, :] - midpointX
        tlengthY = Yzero[0, :] - midpointY

        # angle and corresponding rotatation matrix for
        angle = np.arctan2(np.mean(tlengthX[0:15]), np.mean(tlengthY[0:15]))
        rot_mat = np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        # flattening/formatting X and Y coords for multiplication with rotation matrix
        flat_rot = rot_mat @ np.matrix([Xzero.flatten(), Yzero.flatten()])

        # reshape coordinated back to original shape after rortation
        Xrot = -np.reshape(flat_rot[0, :], [13, frames + 1])
        Yrot = np.reshape(flat_rot[1, :], [13, frames + 1])
        # global varibale for rotated points
        self.Xrot = Xrot
        self.Yrot = Yrot

        # empty matrix for scaled coordinates
        Xscaled = np.zeros((13, frames + 1))
        Yscaled = np.zeros((13, frames + 1))

        # length of trunk
        len = np.mean(np.sqrt((tlengthX[0:15] ** 2) + (tlengthY[0:15] ** 2)))

        # adding scaling factor for trunk length of simaulator
        ltrunk = 0.18
        len = len / ltrunk

        Xscaled[0, :] = Xrot[0, :] / len
        Yscaled[0, :] = Yrot[0, :] / len

        # scaling all points so trunk has length 1
        for i in range(2, 12, 3):
            Xscaled[i - 1, :] = ((Xrot[i - 1, :] - Xrot[0, :]) / len) + Xscaled[0, :]
            Xscaled[i, :] = ((Xrot[i, :] - Xrot[i - 1, :]) / len) + Xscaled[i - 1, :]
            Xscaled[i + 1, :] = ((Xrot[i + 1, :] - Xrot[i, :]) / len) + Xscaled[i, :]

            Yscaled[i - 1, :] = ((Yrot[i - 1, :] - Yrot[0, :]) / len) + Yscaled[0, :]
            Yscaled[i, :] = ((Yrot[i, :] - Yrot[i - 1, :]) / len) + Yscaled[i - 1, :]
            Yscaled[i + 1, :] = ((Yrot[i + 1, :] - Yrot[i, :]) / len) + Yscaled[i, :]

        self.Xscaled = Xscaled
        self.Yscaled = Yscaled

        # print(self.Xscaled[0, -1], self.Xscaled[0, 0])
        # print(self.Yscaled[0, -1], self.Yscaled[0, 0])

        # midpoinp after everything is processed
        self.midpointX = (Xscaled[7, :] + Xscaled[10, :]) / 2
        self.midpointY = (Yscaled[7, :] + Yscaled[10, :]) / 2

        # Ut_angle=

    def trunk_com(self):
        x = self.Xscaled
        y = self.Yscaled

        thet1 = -np.arctan2((x[1, :] - x[4, :]), (y[1, :] - y[4, :]))
        thet2 = -np.arctan2((x[7, :] - x[10, :]), (y[7, :] - y[10, :]))

        # plt.plot(thet2)
        # plt.show()
        # exit()
        h_offset = 0.12

        self.Xcom_h = x[0, :] + (h_offset * np.cos(thet1))
        self.Ycom_h = y[0, :] + (h_offset * np.sin(thet1))

        L_ut = 0.08
        self.Xcom_ut = x[0, :] - (L_ut * np.cos(thet1))
        self.Ycom_ut = y[0, :] - (L_ut * np.sin(thet1))

        L_lt = 0.03
        self.Xcom_lt = self.midpointX - L_lt * np.cos(thet2)
        self.Ycom_lt = self.midpointY - L_lt * np.sin(thet2)

        Fn_u = (1.251 + 0.373 + 0.345) * 9.81
        Fn_l = (1.532 + 0.677 + 0.701) * 9.81
        Fn_h = 0.945 * 9.81

        self.Xcalc = (
            np.multiply(self.Xcom_ut, Fn_u) + np.multiply(self.Xcom_lt, Fn_l) + np.multiply(self.Xcom_h, Fn_h)
        ) / (Fn_u + Fn_l + Fn_h)
        self.Ycalc = (
            np.multiply(self.Ycom_ut, Fn_l) + np.multiply(self.Ycom_lt, Fn_u) + np.multiply(self.Ycom_h, Fn_h)
        ) / (Fn_u + Fn_l + Fn_h)

    def plotCOM(self):
        self.trunk_com()

        plt.subplot(3, 2, 5)
        plt.title("X Lower")
        plt.plot(self.Xcom_lt.T)

        plt.subplot(3, 2, 3)
        plt.title("X Upper")
        plt.plot(self.Xcom_ut.T)

        plt.subplot(3, 2, 1)
        plt.title("X Head")
        plt.plot(self.Xcom_h.T)

        plt.subplot(3, 2, 6)
        plt.title("Y Lower")
        plt.plot(self.Ycom_lt.T)

        plt.subplot(3, 2, 4)
        plt.title("Y Upper")
        plt.plot(self.Ycom_ut.T)

        plt.subplot(3, 2, 2)
        plt.title("Y Head")
        plt.plot(self.Ycom_h.T)

        plt.show()

    def plotskel(self, i):
        i = i + 210
        colors = [
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
        ]

        limbSeq = np.matrix(
            [[1, 0], [2, 1], [3, 2], [4, 0], [5, 4], [6, 5], [7, 0], [8, 7], [9, 8], [10, 0], [11, 10], [12, 11]]
        )
        # limbSeq = np.matrix(
        #     [[1, 0], [2, 1], [3, 2], [4, 0], [5, 4], [6, 5], [7, 10], [8, 7], [9, 8], [13, 0], [11, 10], [12, 11]]
        # )

        x = self.Xscaled[:, i].T
        x = np.asarray(x).reshape(-1)
        x = np.append(x, self.midpointX[i])

        y = self.Yscaled[:, i].T
        y = np.asarray(y).reshape(-1)
        y = np.append(y, self.midpointY[i])

        self.trunk_com()

        plt.cla()

        for p in range(0, 12):

            plt.plot(
                [x[limbSeq[p, 0]], x[limbSeq[p, 1]]],
                [y[limbSeq[p, 0]], y[limbSeq[p, 1]]],
                color=np.array(colors[p]) / 255,
            )
            plt.plot(
                [x[limbSeq[p, 0]], x[limbSeq[p, 1]]],
                [y[limbSeq[p, 0]], y[limbSeq[p, 1]]],
                "o",
            )

        plt.plot(self.Xcom_lt[i], self.Ycom_lt[i], "o", markersize=10, color="k")
        plt.plot(self.Xcom_ut[i], self.Ycom_ut[i], "o", markersize=10, color="c")
        plt.plot(self.Xcom_h[i], self.Ycom_h[i], "o", markersize=10, color="y")

        plt.plot(self.Xcalc[i], self.Ycalc[i], "o", markersize=10, color="r")

        s = "t= " + str(i / 60)

        plt.xlim(-0.25, 0.25)
        plt.ylim(-0.5, 0.25)
        plt.text(0, 0, s)
        plt.grid()
        # plt.show()
        # time.sleep(0.05)
        # exit()

    def plotskel_loop(self):
        fig = plt.figure()
        ani = animation.FuncAnimation(fig, self.plotskel, frames=self.frames - 210, interval=2)
        ani.save(filename=r"C:\Users\franc\Documents\Infant_Sim_data\load tests\side_bend_35.gif", writer="pillow")
        # plt.show()


pp = processPose(r"C:\Users\franc\Documents\Infant_Sim_data\load tests\side_bend_35.pkl")
# pp.plotCOM()
pp.plotskel_loop()
# # pp.trunk_com()
