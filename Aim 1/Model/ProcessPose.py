import numpy as np
import math
import scipy
import pandas as pd
from numpy import radians as radians
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


class processPose:
    def __init__(self, file, start=0):

        # df = pd.read_pickle(file)
        df = pd.read_csv(file)

        # print(df)

        df["x"] = df["x"].astype(float)
        # df["y"] = df["y"].astype(float)
        df["y"] = df["y"].astype(float)

        frames = int(np.max(df.frame))
        self.frames = frames

        self.fps = round(np.mean(df.fps), -1)

        Xparts = np.zeros((18, frames + 1))
        Yparts = np.zeros((18, frames + 1))
        idx = np.zeros((18, frames + 1))
        conf = np.zeros((18, frames + 1))

        self.start = start

        for p in range(18):
            # print(p)
            Xparts[p, :] = df.x[df.part_idx == p]
            Yparts[p, :] = df.y[df.part_idx == p]
            idx[p, :] = p
            conf[p, :] = df.c[df.part_idx == p]

        print(start, frames)

        if start != None:
            Xparts = Xparts[:, start:frames]
            Yparts = Yparts[:, start:frames]
            idx = idx[:, start:frames]
            conf = conf[:, start:frames]

        # if self.fps == 60:
        #     print(Xparts)

        self.Xparts = Xparts
        self.Yparts = Yparts
        self.idx = idx
        self.conf = conf
        self.frames = frames

        # 15 window median filter
        self.Xfilt = scipy.ndimage.median_filter(Xparts, size=[1, 5])
        self.Yfilt = scipy.ndimage.median_filter(Yparts, size=[1, 5])

        # # zeroing coords. to neck position
        # Xzero = self.Xfilt - np.mean(self.Xfilt[0, 0:15])
        # Yzero = self.Yfilt - np.mean(self.Yfilt[0, 0:15])
        # # global variable for zeroed coords.
        # self.Xzero = Xzero
        # self.Yzero = Yzero

        # # midpoint between both hips
        # midpointX = (Xzero[7, :] + Xzero[10, :]) / 2
        # midpointY = (Yzero[7, :] + Yzero[10, :]) / 2

        # # trunk length (from neck to hip midpoint)
        # tlengthX = Xzero[0, :] - midpointX
        # tlengthY = Yzero[0, :] - midpointY

        # # angle and corresponding rotatation matrix for
        # angle = np.arctan2(np.mean(tlengthX[0:15]), np.mean(tlengthY[0:15]))
        # rot_mat = np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        # # flattening/formatting X and Y coords for multiplication with rotation matrix
        # flat_rot = rot_mat @ np.matrix([Xzero.flatten(), Yzero.flatten()])

        # # reshape coordinated back to original shape after rortation
        # Xrot = -np.reshape(flat_rot[0, :], [13, frames + 1])
        # Yrot = np.reshape(flat_rot[1, :], [13, frames + 1])
        # # global varibale for rotated points
        # self.Xrot = Xrot
        # self.Yrot = Yrot

        # # empty matrix for scaled coordinates
        # Xscaled = np.zeros((13, frames + 1))
        # Yscaled = np.zeros((13, frames + 1))

        # # length of trunk
        # len = np.mean(np.sqrt((tlengthX[0:15] ** 2) + (tlengthY[0:15] ** 2)))

        # # adding scaling factor for trunk length of simaulator
        # ltrunk = 0.18
        # len = len / ltrunk

        # Xscaled[0, :] = Xrot[0, :] / len
        # Yscaled[0, :] = Yrot[0, :] / len

        # # scaling all points so trunk has length 1
        # for i in range(2, 12, 3):
        #     Xscaled[i - 1, :] = ((Xrot[i - 1, :] - Xrot[0, :]) / len) + Xscaled[0, :]
        #     Xscaled[i, :] = ((Xrot[i, :] - Xrot[i - 1, :]) / len) + Xscaled[i - 1, :]
        #     Xscaled[i + 1, :] = ((Xrot[i + 1, :] - Xrot[i, :]) / len) + Xscaled[i, :]

        #     Yscaled[i - 1, :] = ((Yrot[i - 1, :] - Yrot[0, :]) / len) + Yscaled[0, :]
        #     Yscaled[i, :] = ((Yrot[i, :] - Yrot[i - 1, :]) / len) + Yscaled[i - 1, :]
        #     Yscaled[i + 1, :] = ((Yrot[i + 1, :] - Yrot[i, :]) / len) + Yscaled[i, :]

        # self.Xscaled = Xscaled
        # self.Yscaled = Yscaled

        # # print(self.Xscaled[0, -1], self.Xscaled[0, 0])
        # # print(self.Yscaled[0, -1], self.Yscaled[0, 0])

        # # midpoinp after everything is processed
        # self.midpointX = (Xscaled[7, :] + Xscaled[10, :]) / 2
        # self.midpointY = (Yscaled[7, :] + Yscaled[10, :]) / 2

        # Ut_angle=

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
            [[1, 0], [2, 1], [3, 2], [4, 0], [5, 4], [6, 5], [7, 10], [8, 7], [9, 8], [13, 0], [11, 10], [12, 11]]
        )

        x = self.Xfilt[:, i].T
        x = np.asarray(x).reshape(-1)
        # x = np.append(x, self.midpointX[i])

        y = self.Yfilt[:, i].T
        y = np.asarray(y).reshape(-1)
        # y = np.append(y, self.midpointY[i])

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

        s = "t= " + str(i / 60)

        plt.xlim(-2500, 2500)
        plt.ylim(-2500, 2500)
        plt.text(0, 0, s)
        plt.grid()
        # plt.show()
        # time.sleep(0.05)
        # exit()

    def plotskel_loop(self):
        fig = plt.figure()
        ani = animation.FuncAnimation(fig, self.plotskel, frames=self.frames - 210, interval=2)
        # ani.save(filename=r"C:\Users\franc\Documents\Infant_Sim_data\load tests\side_bend_35.gif", writer="pillow")
        plt.show()


# pp = processpose(
#     r"C:\Users\franc\Box\Rehab Robotics Lab\Projects\PANDA Gym (# 834084)\Data\Trials\Aim III\833180_252\07-10-2024\Cameras\2024_07_10_833180_252_cam3rec_vid4.csv"
# )
# # pp.plotCOM()
# pp.plotskel_loop()
# # pp.trunk_com()
