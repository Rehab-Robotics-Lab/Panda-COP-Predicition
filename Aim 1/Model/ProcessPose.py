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

        print("Total Frames: ", frames)

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
        Xmean = scipy.ndimage.median_filter(Xparts, size=[1, 5])
        Ymean = scipy.ndimage.median_filter(Yparts, size=[1, 5])

        check_jump = self.find_posejump(Xmean.copy(), Ymean.copy(), view=0)[2]

        # print(self.find_posejump(Xmean, Ymean, view=0)[2])
        if check_jump > 0:
            print("Number of frames with pose jumping Before: ", check_jump)
            x, y = self.Zscore(Xmean, Ymean)
            self.Xfilt, self.Yfilt, tot = self.find_posejump(x, y, view=0)

            print("Number of frames with pose jumping After: ", tot)

        else:
            self.Xfilt, self.Yfilt = Xmean, Ymean
        # self.find_posejump(self.Xfilt, self.Yfilt)

    def find_posejump(self, X, Y, thresh=300, view=1):
        X_diff = np.diff(X, axis=1)
        Y_diff = np.diff(Y, axis=1)

        diff_norm = np.linalg.norm([X_diff, Y_diff], axis=0)

        jump = np.where(diff_norm >= thresh)

        X[jump[0], jump[1] + 1] = np.nan
        Y[jump[0], jump[1] + 1] = np.nan

        if view == 1:
            plt.plot(diff_norm.T)
            plt.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "16", "17"])
            plt.show()

        X_interp, Y_interp = self.interpolate_nans(X, Y)

        return X_interp, Y_interp, len(np.unique(jump[1]))

    def Zscore(self, X, Y, thresh=2.5):
        z_x = np.abs(scipy.stats.zscore(X, axis=1))
        z_y = np.abs(scipy.stats.zscore(Y, axis=1))

        jump = np.where((z_x >= thresh) | (z_y >= thresh))

        X[jump] = np.nan
        Y[jump] = np.nan

        X_z, Y_z = self.interpolate_nans(X, Y)

        return X_z, Y_z

    def interpolate_nans(self, x, y):
        X = pd.DataFrame(x).interpolate(method="linear", axis=1, limit_direction="both").to_numpy()
        Y = pd.DataFrame(y).interpolate(method="linear", axis=1, limit_direction="both").to_numpy()

        return X, Y

    def plotskel(self, i):

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

        x = self.Xfilt[:, i].T
        x = np.asarray(x).reshape(-1)
        # x = np.append(x, self.midpointX[i])

        y = self.Yfilt[:, i].T
        y = np.asarray(y).reshape(-1)
        # y = np.append(y, self.midpointY[i])

        plt.cla()

        for p in range(0, 17):

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

        plt.xlim(np.nanmin(self.Xfilt) - 10, np.nanmax(self.Xfilt) + 10)
        plt.ylim(np.nanmin(self.Yfilt) - 10, np.nanmax(self.Yfilt) + 10)
        plt.text(np.nanmean(self.Xfilt), np.nanmean(self.Yfilt), s)
        plt.grid()
        # plt.show()
        # time.sleep(0.05)
        # exit()

    def plotskel_loop(self):
        fig = plt.figure()
        ani = animation.FuncAnimation(fig, self.plotskel, frames=self.frames - 210, interval=2)
        # ani.save(filename=r"C:\Users\franc\Documents\Infant_Sim_data\load tests\side_bend_35.gif", writer="pillow")
        plt.show()


# pp = processPose(r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\pose\sim_trunk_clothed_cam2_vid5.csv")
# pp.plotskel_loop()
