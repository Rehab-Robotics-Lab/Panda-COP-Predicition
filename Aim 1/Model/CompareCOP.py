from ProcessCOP import processCOP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pandas as pd
from scipy import stats

# from dtaidistance import dtw
# from dtaidistance import dtw_visualisation as dtwvis


class compareCOP:
    def __init__(self, Xcalc, Ycalc, Xreal, Yreal, cam=None):
        if cam == 1:
            n = min(Xcalc.shape[0], Xreal.shape[0])
            self.n = n
            self.Xcalc = Xcalc[0:n]
            self.Ycalc = Ycalc[0:n]

            self.Xreal = Xreal[0:n]
            self.Yreal = Yreal[0:n]
        else:

            self.Xcalc = Xcalc
            self.Ycalc = Ycalc

            self.Xreal = Xreal
            self.Yreal = Yreal

            self.rate = 60

    def comp_XY(self):

        X, Y = self.Xreal, self.Yreal
        n = self.n
        t = (np.linspace(0, n / 30, num=n)).T

        Xcalc = self.Xcalc
        Ycalc = self.Ycalc

        # print(t.shape, X.shape, Xcalc.shape)

        # fig, ax = plt.subplots(2, 1)

        plt.subplot(2, 1, 1)
        plt.plot(t, X - np.mean(X[0:15]))
        plt.plot(t, Xcalc - np.mean(Xcalc[0:15]))
        plt.legend(["Grnd Trth", "Calculated"])
        plt.title("X COP")
        plt.xlabel("Time (s)")
        plt.ylabel("COP X (mm)")
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(t, Y - np.mean(Y[0:15]))
        plt.plot(t, Ycalc - np.mean(Ycalc[0:15]))
        plt.legend(["Grnd Trth", "Calculated"])
        plt.title("Y COP")
        plt.xlabel("Time (s)")
        plt.ylabel("COP Y (mm)")
        plt.grid()

        plt.tight_layout()
        plt.show()

    def ellipse_anim(self, j):
        self.ax1.cla()
        self.ax2.cla()

        # loading zeroing ground truth COP
        Xr = self.Xreal - np.mean(self.Xreal[0:15])
        Yr = self.Yreal - np.mean(self.Yreal[0:15])

        Xc = self.Xcalc - np.mean(self.Xcalc[0:15])
        Yc = self.Ycalc - np.mean(self.Ycalc[0:15])

        xmax, xmin = np.max([np.max(Xc), np.max(Xr)]), np.min([np.min(Xc), np.min(Xr)])
        ymax, ymin = np.max([np.max(Yc), np.max(Yr)]), np.min([np.min(Yc), np.min(Yr)])

        # plt.xlim(xmin - 10, xmax + 10)
        # plt.ylim(ymin - 10, ymax + 10)

        s = "t= " + str(int(j / 30))
        self.ax1.text(0, 0, "%s" % (s), size=20, zorder=1, color="k")

        ##Ellpise plot for ground truth data
        self.ax1.scatter(Xr[j], Yr[j], hatch="x", color="orange")
        self.ax1.plot(Xr[:j], Yr[:j], alpha=0.7)
        self.ax1.grid()
        self.ax1.set_title("Ground Truth")
        self.ax1.set(xlabel="COP X (mm)", ylabel="COP Y (mm)")
        self.ax1.set_xlim(xmin - 10, xmax + 10)
        self.ax1.set_ylim(ymin - 10, ymax + 10)

        ##Ellpise plot for model data
        self.ax2.scatter(Xc[j], Yc[j], hatch="x", color="red")
        self.ax2.plot(Xc[:j], Yc[:j], alpha=0.7)
        self.ax2.grid()
        self.ax2.set_title("Model")
        self.ax2.set(xlabel="COP X (mm)", ylabel="COP Y (mm)")
        self.ax2.set_xlim(xmin - 10, xmax + 10)
        self.ax2.set_ylim(ymin - 10, ymax + 10)

        # plt.xlabel("COP X (mm)")
        # plt.ylabel("COP Y (mm)")

    def xy_anim(self, j):
        self.ax3.cla()
        self.ax4.cla()

        n = self.n
        t = (np.linspace(0, n / 30, num=n)).T

        # loading zeroing ground truth COP
        Xr = self.Xreal - np.mean(self.Xreal[0:15])
        Yr = self.Yreal - np.mean(self.Yreal[0:15])

        Xc = self.Xcalc - np.mean(self.Xcalc[0:15])
        Yc = self.Ycalc - np.mean(self.Ycalc[0:15])

        xmax, xmin = np.max([np.max(Xc), np.max(Xr)]), np.min([np.min(Xc), np.min(Xr)])
        ymax, ymin = np.max([np.max(Yc), np.max(Yr)]), np.min([np.min(Yc), np.min(Yr)])

        # print(t.shape, X.shape, Xcalc.shape)

        # fig, ax = plt.subplots(2, 1)
        s = "t= " + str(int(j / 30))
        self.ax3.text(0, 0, "%s" % (s), size=20, zorder=1, color="k")

        self.ax3.plot(t[:j], Xr[:j], color="orange")
        self.ax3.plot(t[:j], Xc[:j], color="blue")
        self.ax3.legend(["Grnd Trth", "Calculated"])
        self.ax3.set_title("X COP")
        self.ax3.set(xlabel="Time (s)", ylabel="COP X (mm)")
        self.ax3.set_xlim(0, np.max(t) + 1)
        self.ax3.set_ylim(xmin - 10, xmax + 10)
        self.ax3.grid()

        self.ax4.plot(t[:j], Yr[:j], color="orange")
        self.ax4.plot(t[:j], Yc[:j], color="blue")
        self.ax4.legend(["Grnd Trth", "Calculated"])
        self.ax4.set_title("Y COP")
        self.ax4.set(xlabel="Time (s)", ylabel="COP Y (mm)")
        self.ax4.set_xlim(0, np.max(t) + 1)
        self.ax4.set_ylim(ymin - 10, ymax + 10)
        self.ax4.grid()

    def plot_cop_anim(self, start=0):
        fig1, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        fig2, (self.ax3, self.ax4) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
        frms = np.linspace(0, self.n, self.n + 1, dtype=int)
        # print(frms)

        ani1 = animation.FuncAnimation(fig1, self.ellipse_anim, frames=frms[start:], interval=1)
        ani2 = animation.FuncAnimation(fig2, self.xy_anim, frames=frms[start:], interval=1)
        plt.show()

    def comp_XY_robo(self, robot_t, off=0):

        n = len(robot_t)
        t = np.asarray(robot_t).reshape(n, 1)

        X, Y = self.Xreal, self.Yreal
        t_c = np.linspace(0, (len(X) - 1) / self.rate, num=len(X)) + off

        Xcalc = self.Xcalc.reshape(n, 1)
        Ycalc = self.Ycalc.reshape(n, 1)

        # fig, ax = plt.subplots(2, 1)

        plt.subplot(2, 1, 1)
        plt.plot(t_c + t[0], X - np.mean(X[0:5]))
        plt.plot(t, Xcalc - np.mean(Xcalc[0:5]))
        plt.legend(["Grnd Trth", "Calculated"])
        plt.title("X COP")
        plt.xlabel("Time (s)")
        plt.ylabel("COP X (mm)")
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(t_c + t[0], Y - np.mean(Y[0:5]))
        plt.plot(t, Ycalc - np.mean(Ycalc[0:5]))
        plt.legend(["Grnd Trth", "Calculated"])
        plt.title("Y COP")
        plt.xlabel("Time (s)")
        plt.ylabel("COP Y (mm)")
        plt.grid()

        plt.tight_layout()
        plt.show()

    def comp_XY(self):

        X, Y = self.Xreal, self.Yreal
        n = self.n
        t = (np.linspace(0, n / 30, num=n)).T

        Xcalc = self.Xcalc
        Ycalc = self.Ycalc

        # print(t.shape, X.shape, Xcalc.shape)

        # fig, ax = plt.subplots(2, 1)

        plt.subplot(2, 1, 1)
        plt.plot(t, X - np.mean(X[0:15]))
        plt.plot(t, Xcalc - np.mean(Xcalc[0:15]))
        plt.legend(["Grnd Trth", "Calculated"])
        plt.title("X COP")
        plt.xlabel("Time (s)")
        plt.ylabel("COP X (mm)")
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(t, Y - np.mean(Y[0:15]))
        plt.plot(t, Ycalc - np.mean(Ycalc[0:15]))
        plt.legend(["Grnd Trth", "Calculated"])
        plt.title("Y COP")
        plt.xlabel("Time (s)")
        plt.ylabel("COP Y (mm)")
        plt.grid()

        plt.tight_layout()
        plt.show()

    def comp_ellipse(self):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

        # loading zeroing ground truth COP
        Xr = (self.Xreal - np.mean(self.Xreal[0:15])).T
        Yr = (self.Yreal - np.mean(self.Yreal[0:15])).T

        # loading zeroing model COP
        Xcalc = self.Xcalc
        Ycalc = self.Ycalc

        Xc = Xcalc - np.mean(Xcalc[0:15])
        Yc = Ycalc - np.mean(Ycalc[0:15])

        xmax, xmin = np.max([np.max(Xc), np.max(Xr)]), np.min([np.min(Xc), np.min(Xr)])
        ymax, ymin = np.max([np.max(Yc), np.max(Yr)]), np.min([np.min(Yc), np.min(Yr)])

        plt.xlim(xmin - 10, xmax + 10)
        plt.ylim(ymin - 10, ymax + 10)

        ##Ellpise plot for ground truth data
        self.confidence_ellipse(Xr, Yr, ax1, edgecolor="red")
        ax1.scatter(Xr, Yr, s=0.5)
        ax1.grid()
        ax1.set_title("Ground Truth")
        ax1.set(xlabel="COP X (mm)", ylabel="COP Y (mm)")

        ##Ellpise plot for model data
        self.confidence_ellipse(Xc, Yc, ax2, edgecolor="red")
        ax2.scatter([Xc], [Yc], s=0.5)
        ax2.grid()
        # plt.xlabel("COP X (mm)")
        # plt.ylabel("COP Y (mm)")
        ax2.set_title("Model")
        ax2.set(xlabel="COP X (mm)", ylabel="COP Y (mm)")

        # plt.xlabel("COP X (mm)")
        # plt.ylabel("COP Y (mm)")

        plt.show()

    def confidence_ellipse(self, x, y, ax, n_std=2.0, facecolor="none", **kwargs):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.

        Parameters
        ----------
        x, y : array-like, shape (n, )
            Input data.

        ax : matplotlib.axes.Axes
            The Axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`

        Returns
        -------
        matplotlib.patches.Ellipse
        """
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensional dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

        # Calculating the standard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the standard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)

        return ax.add_patch(ellipse)

    def ellipse_area(self, x, y, n_std=2.0):
        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensional dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2)

        # Calculating the standard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the standard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

        ellipse.set_transform(transf)

        area = ellipse.width * ellipse.height * np.pi
        return area

    def dtw(self):
        # loading zeroing ground truth COP
        Xr = self.Xreal - np.mean(self.Xreal)
        Yr = self.Yreal - np.mean(self.Yreal)

        # loading zeroing model COP
        Xcalc = self.Xcalc
        Ycalc = self.Ycalc

        Xc = Xcalc - np.mean(Xcalc)
        Yc = Ycalc - np.mean(Ycalc)

        win = 1

        # print(np.shape(Xc), np.shape(Xr))
        # pathsX = dtw.warping_path(Xc.T, Xr, window=10)
        # best_pathX = dtw.best_path(pathsX)
        dX, pathX = dtw.warp(Xc.T, Xr, window=win)

        # pathsY = dtw.warping_path(Yc.T, Yr, window=5)
        # best_pathY = dtw.best_path(pathsY)
        dY, pathY = dtw.warp(Yc.T, Yr, window=win)

        self.Xcalc_sync = dX
        self.Ycalc_sync = dY

        plt.subplot(2, 1, 1)
        plt.plot(Xr)
        plt.plot(dX)

        plt.subplot(2, 1, 2)
        plt.plot(Yr)
        plt.plot(dY)

        plt.show()

    def mse(self):
        # Xr = self.Xreal - np.mean(self.Xreal)
        # Yr = self.Yreal - np.mean(self.Yreal)

        # dX = self.Xcalc_sync
        # dY = self.Ycalc_sync

        Xr = self.Xreal - np.mean(self.Xreal[0:15])
        Yr = self.Yreal - np.mean(self.Yreal[0:15])

        dX = self.Xcalc - np.mean(self.Xcalc[0:15])
        dY = self.Ycalc - np.mean(self.Ycalc[0:15])

        mseX = np.mean((Xr - dX) ** 2)
        mseY = np.mean((Yr - dY) ** 2)

        return mseX, mseY

    def mae(self):
        Xr = self.Xreal - np.mean(self.Xreal[0:15])
        Yr = self.Yreal - np.mean(self.Yreal[0:15])

        dX = self.Xcalc - np.mean(self.Xcalc[0:15])
        dY = self.Ycalc - np.mean(self.Ycalc[0:15])

        maeX = np.mean(np.abs(Xr - dX))
        maeY = np.mean(np.abs(Yr - dY))

        return maeX, maeY

    def pearson_corr(self):
        Xr = np.atleast_2d(self.Xreal - np.mean(self.Xreal[0:15])).T
        Yr = np.atleast_2d(self.Yreal - np.mean(self.Yreal[0:15])).T

        dX = self.Xcalc - np.mean(self.Xcalc[0:15])
        dY = self.Ycalc - np.mean(self.Ycalc[0:15])

        print(np.shape(Xr), np.shape(dX))

        val_X, p_X = stats.pearsonr(Xr, dX)
        val_Y, p_Y = stats.pearsonr(Yr, dY)

        return val_X, p_X, val_Y, p_Y

    def excursion(self, x):
        excur = np.max(x) - np.min(x)

        return excur

    def std(self, x):

        return np.std(x)

    def path_len(self, x, y):
        # n = len(x)

        x = x - np.mean(x)
        y = y - np.mean(y)

        curX = abs(x)
        curY = abs(y)

        copMag = np.sqrt(np.power(curX, 2) + np.power(curY, 2))

        pathLen = np.zeros(len(copMag) - 1)

        # for i = 2:1:size(pathLen) - 1
        for i in range(len(pathLen) - 1):
            pathLen[i] = np.sqrt((curX[i] - curX[i - 1]) ** 2 + (curY[i] - curY[i - 1]) ** 2)

        avg_pathLen = sum(pathLen) / (len(copMag))

        return avg_pathLen

    def diff_metric(self, dtw=0):
        if dtw == 1:
            self.dtw()

        maeX, maeY = self.mae()
        # mseX, mseY = self.mse()
        val_x, p_x, val_y, p_y = self.pearson_corr()

        data = {
            "X": [maeX, val_x, p_x],
            "Y": [maeY, val_y, p_y],
        }

        df = pd.DataFrame(data, index=["MAE", "Pearson", "P-value"])

        return df

    def metrics(self):
        # n = len(x)

        Xc, Yc = self.Xcalc, self.Ycalc
        Xr, Yr = self.Xreal, self.Yreal

        data = {
            "Ground Truth": [
                self.ellipse_area(Xr, Yr),
                self.excursion(Xr),
                self.excursion(Yr),
                self.std(Xr),
                self.std(Yr),
                self.path_len(Xr, Yr),
            ],
            "Model": [
                self.ellipse_area(Xc, Yc),
                self.excursion(Xc),
                self.excursion(Yc),
                self.std(Xc),
                self.std(Yc),
                self.path_len(Xc.T, Yc.T),
            ],
        }

        df = pd.DataFrame(data, index=["Area", "Excur X", "Excur Y", "Std X", "Std Y", "Path Len"])

        return df
