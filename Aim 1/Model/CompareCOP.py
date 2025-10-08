from ProcessCOP import processCOP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pandas as pd
from scipy import constants

# from dtaidistance import dtw
# from dtaidistance import dtw_visualisation as dtwvis


class compareCOP:
    def __init__(self, Xcalc, Ycalc, Xreal, Yreal):
        self.Xcalc = Xcalc
        self.Ycalc = Ycalc

        self.Xreal = Xreal
        self.Yreal = Yreal

        self.rate = 60

    def comp_XY(self, robot_t, off=0):

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

    def comp_ellipse(self):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

        # loading zeroing ground truth COP
        Xr = (self.Xreal - np.mean(self.Xreal)).T
        Yr = (self.Yreal - np.mean(self.Yreal)).T

        # loading zeroing model COP
        Xcalc = self.Xcalc
        Ycalc = self.Ycalc

        Xc = Xcalc - np.mean(Xcalc)
        Yc = Ycalc - np.mean(Ycalc)

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
        Xr = self.Xreal - np.mean(self.Xreal)
        Yr = self.Yreal - np.mean(self.Yreal)

        dX = self.Xcalc_sync
        dY = self.Ycalc_sync

        mseX = np.mean((Xr - dX) ** 2)
        mseY = np.mean((Yr - dY) ** 2)

        return mseX, mseY

    def mae(self):
        Xr = self.Xreal - np.mean(self.Xreal)
        Yr = self.Yreal - np.mean(self.Yreal)

        dX = self.Xcalc_sync
        dY = self.Ycalc_sync

        maeX = np.mean(np.abs(Xr - dX))
        maeY = np.mean(np.abs(Yr - dY))

        return maeX, maeY

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

    def diff_metric(self):
        self.dtw()
        maeX, maeY = self.mae()
        mseX, mseY = self.mse()

        data = {
            "X": [
                maeX,
                mseX,
            ],
            "Y": [
                maeY,
                mseY,
            ],
        }

        df = pd.DataFrame(data, index=["MAE", "MSE"])

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
