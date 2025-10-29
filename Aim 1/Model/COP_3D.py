# from Inv_dynamics import inv_dynamics
from ProcessCOP import processCOP
from CompareCOP import compareCOP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import scipy
from ProcessPose_3D import processpose


class sim_COP:
    # mass=[2X1], length=[2x1], I=[2x3], theta=[4,t]
    def __init__(self):

        posefile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Cameras\sim_cam_2_6_vid_3.csv"

        # LOADING FILE WITH COP VALUES
        cop_file = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\2025-6-11_1_55_Sim_each_2_3.csv"
        # self.rate = 60
        # # cop object
        self.COP = processCOP(cop_file, 60)

        pose = processpose(posefile)
        pose.IK_init()
        self.frames = pose.frames

        # self.pose.ID_init()

        ##INITIALIZING PARAMETERS

        # Arm paramters
        # lengths
        uarm_len = (np.mean(pose.get_len(2, 3)) + np.mean(pose.get_len(5, 6))) / 2
        larm_len = (np.mean(pose.get_len(3, 4)) + np.mean(pose.get_len(6, 7))) / 2

        # radius
        uarm_r = 0.20
        larm_r = 0.15
        # mass
        l_uarm_m = 0.2
        l_larm_m = 0.424 - l_uarm_m - 0.072
        r_uarm_m = 0.22
        r_larm_m = 0.449 - r_uarm_m - 0.075
        # inertia
        l_uarm_I = self.I(uarm_len, l_uarm_m, uarm_r)
        l_larm_I = self.I(larm_len, l_larm_m, larm_r)
        r_uarm_I = self.I(uarm_len, r_uarm_m, uarm_r)
        r_larm_I = self.I(larm_len, r_larm_m, larm_r)

        # storing left arm parameters
        self.larm_m = np.array([l_uarm_m, l_larm_m])
        self.larm_len = np.array([uarm_len, larm_len])
        self.larm_I = np.matrix([l_uarm_I, l_larm_I])
        # storing right arm parameters
        self.rarm_m = np.array([r_uarm_m, r_larm_m])
        self.rarm_len = self.larm_len
        self.rarm_I = np.matrix([r_larm_I, r_uarm_I])

        # Leg paramters
        # lengths
        uleg_len = (np.mean(pose.get_len(8, 9)) + np.mean(pose.get_len(11, 12))) / 2
        lleg_len = (np.mean(pose.get_len(9, 10)) + np.mean(pose.get_len(12, 13))) / 2
        # radius
        uleg_r = 0.65
        lleg_r = 0.4
        # mass
        l_uleg_m = 0.4
        l_lleg_m = 0.805 - l_uleg_m
        r_uleg_m = 0.4
        r_lleg_m = 0.823 - r_uleg_m
        # inertia
        l_uleg_I = self.I(uleg_len, l_uleg_m, uleg_r)
        l_lleg_I = self.I(lleg_len, l_lleg_m, lleg_r)
        r_uleg_I = self.I(uleg_len, r_uleg_m, uleg_r)
        r_lleg_I = self.I(lleg_len, r_lleg_m, lleg_r)

        # storing left leg parameters
        self.lleg_m = np.array([l_uleg_m, l_lleg_m])
        self.lleg_len = np.array([uleg_len, lleg_len])
        self.lleg_I = np.matrix([l_uleg_I, l_lleg_I])

        # storing right leg parameters
        self.rleg_m = np.array([r_uleg_m, r_lleg_m])
        self.rleg_len = self.lleg_len
        self.rleg_I = np.matrix([r_lleg_I, r_uleg_I])

        # Upper Trunk parameters
        self.utrunk_l = 0.15
        self.utrunk_w = np.mean(pose.get_len(2, 5))
        # self.utrunk_w = 0.18
        self.utrunk_h = 0.08
        self.utrunk_m = 1.033 - (l_uarm_m + l_larm_m + r_uarm_m + r_larm_m)

        # Lower Trunk parameters
        self.ltrunk_l = 0.11
        self.ltrunk_w = np.mean(pose.get_len(8, 11))
        # self.ltrunk_w = 0.18
        self.ltrunk_h = 0.08
        self.ltrunk_m = 2.641 - (l_uleg_m + l_lleg_m + r_uleg_m + r_lleg_m)

        self.pose = pose

    def plot_cop_3d(self, j):
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

        x = self.pose.X[j, :] * 1000
        y = self.pose.Y[j, :] * 1000
        z = self.pose.Z[j, :] * 1000

        Xcalc, Ycalc = self.Xcalc.T, self.Ycalc.T

        Xreal, Yreal = self.COP.Xfilt[::2], self.COP.Yfilt[::2]

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

        # self.ax.scatter(Xreal[j] / 1000, Yreal[j] / 1000, 0, marker="o")

        # print(self.Xcop[:, j])
        # print("Fn Total ")
        # print(self.Fn_tot[j])
        # print(self.Xcop[:, j] / self.Fn_tot[j])
        # print("Fn Individual ")
        # print(self.Fn[:, j])
        # print(self.Xcop[:, j] / self.Fn[:, j])
        # quit()

        for i in [0, 1, 2, 3, 4, 5, 6]:
            Fn = self.Fn[i, j]

            Xcop = np.divide(self.Xcop[i, j], Fn)
            Ycop = np.divide(self.Ycop[i, j], Fn)
            Fn_norm = Fn / self.Fn_tot[j] * 1000
            self.ax.quiver(Xcop, Ycop, -Fn_norm, 0, 0, Fn_norm, color="black")
        # print(Fn)
        # print(Xcop, Ycop, -Fn, Xcop, Ycop, 0)
        Fn_tot = 1000

        self.ax.quiver(Xcalc[j], Ycalc[j], -Fn_tot, 0, 0, Fn_tot, color="red")

        Fv = self.Flarm[:, j] / self.Fn_tot[j] * 1000
        self.ax = self.plotF(j, Fv, 5, self.ax)

        # Tv = self.Tlarm[:, j] / np.max(self.Tlarm)
        # self.ax = self.plotT(j, Tv, 5, self.ax)

        s = "t= " + str(int(j / 60))
        # s2 = "s= " + str(round(j / 30))

        self.ax.text(0.3, -0.3, 0, "%s" % (s), size=20, zorder=1, color="k")

        # self.ax.set_zlim3d(0, 600)
        # self.ax.set_ylim3d(-100, 50)
        # self.ax.set_xlim3d(0, 100)

        self.ax.set_zlim3d(0, 600)
        self.ax.set_ylim3d(-500, 200)
        self.ax.set_xlim3d(-300, 300)
        plt.grid()

    def plot_cop_anim(self, start=0):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")

        frms = np.linspace(0, self.frames, self.frames + 1, dtype=int)
        # print(frms)

        ani = animation.FuncAnimation(self.fig, self.plot_cop_3d, frames=frms[start:], interval=1)
        plt.show()

    def plotT(self, i, Tvec, k, ax, scale=1000):
        origin = np.matrix([self.pose.X[i, k], self.pose.Y[i, k], self.pose.Z[i, k]])
        r1 = Tvec[0] * scale / 10
        r2 = Tvec[1] * scale / 10
        r3 = Tvec[2] * scale / 10

        x, y, z = origin[0, 0] * scale, origin[0, 1] * scale, origin[0, 2] * scale

        theta = np.linspace(0, 2 * np.pi, 100)

        x1, y1, z1 = np.zeros_like(theta) + x, np.cos(theta) * r1 + y, np.sin(theta) * r1 + z
        x2, y2, z2 = np.cos(theta) * r2 + x, np.zeros_like(theta) + y, np.sin(theta) * r2 + z
        x3, y3, z3 = np.cos(theta) * r3 + x, np.sin(theta) * r3 + y, np.zeros_like(theta) + z

        ax.plot(x1, y1, z1, color=[1, 0, 0])
        ax.plot(x2, y2, z2, color="green")
        ax.plot(x3, y3, z3, color=[0, 0, 1])

        return ax

    def plotF(self, i, Fvec, k, ax, scale=100):
        origin = np.matrix([self.pose.X[i, k], self.pose.Y[i, k], self.pose.Z[i, k]])
        dx = Fvec[0] * scale
        dy = Fvec[1] * scale
        dz = Fvec[2] * scale

        x, y, z = origin[0, 0] * scale, origin[0, 1] * scale, origin[0, 2] * scale

        ax.quiver(x, y, z, dx, 0, 0, colors=[1, 0, 0])
        ax.quiver(x, y, z, 0, dy, 0, colors="green")
        ax.quiver(x, y, z, 0, 0, dz, colors=[0, 0, 1])

        return ax

    def I(self, l, m, r):

        I1 = 1 / 2 * (m * r**2)
        I2 = (1 / 4 * (m * r**2)) + (1 / 3 * (m * l**2))
        # I2 = 1 / 12 * (m * (3 * r**2 + l**2))

        return np.array([I2, I1, I2])

    def plot_ID(self, T, F):
        plt.subplot(2, 1, 1)
        plt.plot(T.T)
        plt.title("Torque")
        plt.legend(["Tx", "Ty", "Tz"])

        plt.subplot(2, 1, 2)
        plt.plot(F.T)
        plt.title("Force")
        plt.legend(["Fx", "Fy", "Fz"])

        plt.show()

    def COP_upper(self):
        g = 9.81
        pose = self.pose

        # right arm dynamics
        T_ra = np.array([pose.thet1_ra, pose.thet2_ra, pose.thet3_ra, pose.thet4_ra])[:, 0, :]
        Trarm, Frarm = pose.inv_dynamics(T_ra, self.rarm_len, self.rarm_m, self.rarm_I)
        rarm_check = np.ravel((np.rad2deg(pose.thet3_ra) < -45))

        Trarm[:, rarm_check] = 0
        Frarm[:, rarm_check] = 0
        # Frarm[2, rarm_check] = -(np.sum(self.rarm_m) - 0.310) * g

        # self.plot_ID(Trarm, Frarm)

        # Wrinting dynamic terms for right arm
        T1x = Trarm[0, :]
        T1y = Trarm[1, :]

        F1x = Frarm[0, :]
        F1y = Frarm[1, :]
        F1z = Frarm[2, :]

        # left arm dynamics
        T_la = np.array([pose.thet1_la, pose.thet2_la, pose.thet3_la, pose.thet4_la])[:, 0, :]
        Tlarm, Flarm = pose.inv_dynamics(T_la, self.larm_len, self.larm_m, self.larm_I)
        larm_check = np.ravel(np.rad2deg(pose.thet3_la) > 50)
        Tlarm[:, larm_check] = 0
        Flarm[:, larm_check] = 0
        self.Tlarm, self.Flarm = Tlarm, Flarm
        # Flarm[2, larm_check] = -(np.sum(self.larm_m) - 0.367) * g

        # self.plot_ID(Tlarm, Flarm)

        # Wrinting dynamic terms for left arm
        T2x = Tlarm[0, :]
        T2y = Tlarm[1, :]

        F2x = Flarm[0, :]
        F2y = Flarm[1, :]
        F2z = Flarm[2, :]

        # Upper trunk parameters
        l = self.utrunk_l
        w = self.utrunk_w
        M = self.utrunk_m
        h = self.utrunk_h

        # X_calc = (F2 * 0.5 * w) / (F2 + g * M)
        # Y_calc = ((F2 * l / 2) - T2) / (F2 + g * M)
        Xlow, Ylow, FnLow, Tx, Ty = self.COP_lower()

        # X_calc = ((T1y - T2y - Tt) + ((F1x - F2x) * h * 0.5) + ((F2z - F1z) * w * 0.5)) / (F1z + F2z + (g * M))
        # Y_calc = (((F1z + F2z) * l * 0.5) - ((F1y + F2y) * h * 0.5) - (T1x + T2x)) / (F1z + F2z + (g * M))

        # plt.plot(Tx)
        # plt.plot(Ty)
        # plt.legend(["Tx", "Ty"])
        # plt.show()
        Txx = T1x + T2x + Tx
        Tyy = T1y + T2y + Ty
        # plt.plot(Txx)
        # plt.plot(Tyy)
        # plt.legend(["Txx", "Tyy"])
        # plt.show()

        mg = g * M
        F12_x = F1x + F2x
        F12_y = F1y + F2y
        F12_z = F1z + F2z

        dx = w / 2
        dy = l / 4
        dz = h / 2

        # X_calc = np.divide((Tyy - (F12_x * dz) + ((F1z - F2z) * dx)), (mg - F12_z))
        # Y_calc = np.divide((Txx - (F12_y * dz) + (F12_z * dy)), (F12_z - mg))

        # X_calc = ((T1y + T2y + Ty) - ((F1x + F2x) * h * 0.5) + ((F1z - F2z) * w * 0.5)) / ((g * M) - (F1z + F2z))
        # Y_calc = ((T1x + T2x + Tx) + ((F1z + F2z) * l * 0.5) - ((F1y + F2y) * h * 0.5)) / ((F1z + F2z) - (g * M))

        X_calc = np.divide((Tyy + ((F1z - F2z) * dx) + (F12_x * dz)), (mg - F12_z))
        Y_calc = np.divide((Txx - (F12_y * dz) + (F12_z * dy)), (F12_z - mg))

        # plt.plot((T1y + T2y + Ty))
        # plt.plot(((F1z - F2z) * w * 0.5))
        # plt.legend(["Ty", "Fz"])
        # plt.show()

        # X_calc = ((T1y + T2y + Tt) + ((F1z - F2z) * w * 0.5)) / (-F1z - F2z + (g * M))
        # Y_calc = ((T1x - T2x) + ((F1z + F2z) * l * 0.5)) / (F1z + F2z - (g * M))

        mids_X = (pose.X[:, 2] + pose.X[:, 5]) * 0.5
        mids_Y = (pose.Y[:, 2] + pose.Y[:, 5]) * 0.5

        # plt.plot(mids_Y - dy)
        # plt.plot(Y_calc)
        # plt.plot(mids_Y - dy + Y_calc)
        # plt.plot(mids_X)
        # plt.plot(X_calc)
        # plt.plot(X_calc + mids_X + 0.01)
        # plt.grid()
        # plt.legend(["Offset COP", "COP change", "Final COP"])
        # plt.show()
        # quit()

        X_calc = mids_X + X_calc
        Y_calc = mids_Y - dy + Y_calc

        Fn_up = (g * M * np.ones_like(X_calc)) - F1z - F2z

        Xup = np.multiply(X_calc, Fn_up)
        Yup = np.multiply(Y_calc, Fn_up)

        self.rarm_check, self.larm_check = rarm_check, larm_check

        return Xup, Yup, Fn_up, Xlow, Ylow, FnLow

    def COP_lower(self):
        pose = self.pose
        g = 9.81

        # right leg dynamics
        T_rl = np.array([pose.thet1_rl, pose.thet2_rl, pose.thet3_rl, pose.thet4_rl])[:, 0, :]
        Trleg, Frleg = pose.inv_dynamics(T_rl, self.rleg_len, self.rleg_m, self.rleg_I)

        # zeroing force and torque when leg is at rest
        rleg_check = np.ravel(np.rad2deg(pose.thet3_rl) < 50)
        Trleg[:, rleg_check] = 0
        Frleg[:, rleg_check] = 0
        # Frleg[2, rleg_check] = -(np.sum(self.rleg_m) - 0.340) * g

        # self.plot_ID(Trleg, Frleg)

        # right leg dynamic terms
        T3x = Trleg[0, :]
        T3y = Trleg[1, :]

        F3x = Frleg[0, :]
        F3y = Frleg[1, :]
        F3z = Frleg[2, :]

        # rotating torque and force values by 30 degrees
        # Tlleg[0, :] = self.lleg_Ld
        # left leg dynamics
        T_ll = np.array([pose.thet1_ll, pose.thet2_ll, pose.thet3_ll, pose.thet4_ll])[:, 0, :]
        Tlleg, Flleg = pose.inv_dynamics(T_ll, self.lleg_len, self.lleg_m, self.lleg_I)
        lleg_check = np.ravel(np.rad2deg(pose.thet3_ll) > -50)

        # zeroing force and torque when leg is at rest
        Tlleg[:, lleg_check] = 0
        Flleg[:, lleg_check] = 0
        # Flleg[2, lleg_check] = -(np.sum(self.lleg_m) - 0.271) * g

        # self.plot_ID(Tlleg, Flleg)

        # left leg dynamic terms
        T4x = Tlleg[0, :]
        T4y = Tlleg[1, :]

        F4x = Flleg[0, :]
        F4y = Flleg[1, :]
        F4z = Flleg[2, :]

        # plt.plot(rleg_check.T)
        # plt.plot(lleg_check.T)
        # plt.show()
        # exit()

        # lower trunk parameters
        l = self.ltrunk_l
        w = self.ltrunk_w
        M = self.ltrunk_m
        h = self.ltrunk_h

        X_calc = (pose.X[:, 8] + pose.X[:, 11]) * 0.5
        Y_calc = (pose.Y[:, 8] + pose.Y[:, 11]) * 0.5 - 0.01
        # Y_calc = ((T3x - T4x)) / (F3z + F4z - (g * M))

        mg = g * M
        F34_x = F3x + F4x
        F34_y = F3y + F4y
        F34_z = F3z + F4z

        T34_x = T3x + T4x
        T34_y = T3y + T4y

        dx = w / 2
        dy = l / 4
        dz = h / 2

        Tx = T34_x - (F34_z * dy) - (F34_y * dz)
        Ty = T34_y + ((F3z - F4z) * dx) + (F34_x * dz)

        # Ty = (T4y + T3y) + ((F3z - F4z) * w * 0.5) + ((F3x + F4x) * h * 0.5)
        # Tx = (T3x + T4x) - ((F3y + F4y) * h * 0.5) - ((F3z + F4z) * l * 0.5)

        Fn = (g * M * np.ones_like(F3z)) - F3z - F4z

        # plt.plot(Tx.T)
        # plt.plot(Ty.T)
        # plt.plot(Fn.T)
        # plt.legend(["Tx", "Ty", "Fn"])
        # plt.show()
        # exit()

        self.rleg_check, self.lleg_check = rleg_check, lleg_check

        return np.multiply(X_calc, Fn), np.multiply(Y_calc, Fn), Fn, Tx, Ty

    def calc_COP(self):

        X_up, Y_up, Fn_up, X_low, Y_low, Fn_low = self.COP_upper()

        # reshaping to make sure all dimenesions are [1,n]
        n = len(self.lleg_check)

        lleg_check = self.lleg_check
        rleg_check = self.rleg_check
        larm_check = self.larm_check
        rarm_check = self.rarm_check

        f = 1

        Fn_head = np.ones_like(Fn_up) * 1.033 * 9.81
        X_head = np.multiply(self.pose.X[:, 0], Fn_head)
        Y_head = np.multiply(self.pose.Y[:, 0], Fn_head)

        Fn_lleg = np.sum(self.lleg_m) * 9.81 * lleg_check * f
        X_lleg = np.multiply(self.pose.X[:, 13], Fn_lleg)
        Y_lleg = np.multiply(self.pose.Y[:, 13], Fn_lleg)

        Fn_rleg = np.sum(self.rleg_m) * 9.81 * rleg_check * f
        X_rleg = np.multiply(self.pose.X[:, 10], Fn_rleg)
        Y_rleg = np.multiply(self.pose.Y[:, 10], Fn_rleg)

        # Fn_larm = 0.367 * 9.81 * larm_check * f
        Fn_larm = np.sum(self.larm_m) * 9.81 * larm_check * f
        X_larm = np.multiply(self.pose.X[:, 6], Fn_larm)
        Y_larm = np.multiply(self.pose.Y[:, 6], Fn_larm)

        # Fn_rarm = 0.310 * 9.81 * rarm_check * f
        Fn_rarm = np.sum(self.rarm_m) * 9.81 * rarm_check * f
        X_rarm = np.multiply(self.pose.X[:, 3], Fn_rarm)
        Y_rarm = np.multiply(self.pose.Y[:, 3], Fn_rarm)

        # print(Fn_rarm, Fn_larm)

        Fn_tot = Fn_up + Fn_low + Fn_lleg + Fn_rleg + Fn_larm + Fn_rarm + Fn_head

        self.Xcalc = (X_low + X_up + X_lleg + X_rleg + X_larm + X_rarm + X_head) / Fn_tot * 1000

        self.Ycalc = (Y_low + Y_up + Y_lleg + Y_rleg + Y_larm + Y_rarm + Y_head) / Fn_tot * 1000

        self.Fn = np.array([Fn_low, Fn_up, Fn_lleg, Fn_rleg, Fn_larm, Fn_rarm, Fn_head])
        self.Xcop = np.array([X_low, X_up, X_lleg, X_rleg, X_larm, X_rarm, X_head]) * 1000
        self.Ycop = np.array([Y_low, Y_up, Y_lleg, Y_rleg, Y_larm, Y_rarm, Y_head]) * 1000

        self.Fn_tot = Fn_tot

        # print(np.shape(self.Fn), np.shape(self.Xcop), np.shape(self.Ycop))

        # plt.plot(X_up / Fn_tot)
        # plt.plot(X_low / Fn_tot)
        # plt.plot(X_lleg / Fn_tot)
        # plt.plot(X_rleg / Fn_tot)
        # plt.plot(X_larm / Fn_tot)
        # plt.plot(X_rarm / Fn_tot)
        # plt.plot(X_head / Fn_tot)
        # plt.plot(self.Xcalc)

        # plt.plot(Fn_up)
        # plt.plot(Fn_low)
        # plt.plot(Fn_lleg)
        # plt.plot(Fn_rleg)
        # plt.plot(Fn_larm)
        # plt.plot(Fn_rarm)
        # plt.plot(Fn_head)
        # plt.plot(Fn_tot)

        # plt.legend(["Upper", "Lower", "L_leg", "R_Leg", "L_arm", "R_arm", "Head", "Tot"])
        # plt.grid()
        # plt.show()

    def compare_COP(self):
        # cc = compareCOP(self.Xcalc[0, 5:-1], self.Ycalc[0, 5:-1], self.COP.Xfilt[5:-1], self.COP.Yfilt[5:-1])
        # cc.comp_XY(self.robot_t[5:-1], -0.2)
        # cc.comp_ellipse()
        # print(cc.metrics())
        # print(cc.diff_metric())
        Xcalc = self.Xcalc.T
        Ycalc = self.Ycalc.T

        # COP_gc = processCOP(file, 60)

        Xreal = self.COP.Xfilt[::2]
        Yreal = self.COP.Yfilt[::2]

        # cc = compareCOP(COP_X, COP_Y, X_gc, Y_gc)

        # fig, ax = plt.subplots(2, 1)
        delay = 90
        Xreal, Yreal = Xreal[delay:], Yreal[delay:]
        Xcalc, Ycalc = Xcalc[delay:], Ycalc[delay:]

        # Xreal, Yreal = Xreal[delay:2500], Yreal[delay:2500]
        # Xcalc, Ycalc = Xcalc[delay:2500], Ycalc[delay:2500]

        win = 5

        Xcalc = scipy.ndimage.median_filter(Xcalc, win)
        Ycalc = scipy.ndimage.median_filter(Ycalc, win)

        # print(np.mean(Xcalc) - np.mean(Xreal))
        # print(np.mean(Ycalc) - np.mean(Yreal))

        plt.subplot(2, 1, 1)
        # plt.plot(Xcalc - np.mean(Xcalc[0:30]))
        # plt.plot(Xreal - np.mean(Xreal[0:30]))
        # plt.legend(["Calculated", "Grnd Trth"])
        plt.plot(Xreal - np.mean(Xreal[0:30]))
        plt.plot(Xcalc - np.mean(Xcalc[0:30]))
        plt.legend(["Grnd Trth", "Calculated"])
        plt.title("X COP")
        plt.xlabel("Time (s)")
        plt.ylabel("COP X (mm)")
        plt.grid()

        plt.subplot(2, 1, 2)
        # plt.plot(Ycalc - np.mean(Ycalc[0:30]))
        # plt.plot(Yreal - np.mean(Yreal[0:30]))
        # plt.legend(["Calculated", "Grnd Trth"])
        plt.plot(Yreal - np.mean(Yreal[0:30]))
        plt.plot(Ycalc - np.mean(Ycalc[0:30]))
        plt.legend(["Grnd Trth", "Calculated"])
        plt.title("Y COP")
        plt.xlabel("Time (s)")
        plt.ylabel("COP Y (mm)")
        plt.grid()

        plt.tight_layout()
        plt.show()


test = sim_COP()
test.calc_COP()
print("calculation done")
test.plot_cop_anim()
# test.compare_COP()
