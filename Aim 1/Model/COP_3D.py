# from Inv_dynamics import inv_dynamics
from ProcessCOP import processCOP
from CompareCOP import compareCOP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import scipy
from scipy.spatial.transform import Rotation as R

from ProcessPose_3D import processpose


class sim_COP:
    def __init__(self):

        # posefile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\3D Pose Test\sim_vid3_cams_2_4_both.csv"
        # posefile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\3D Pose Test\sim_vid4_cams_2_4_both.csv"
        # posefile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\3D Pose Test\sim_trunk_clothed_vid3_cams_2_4_both.csv"
        # posefile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\3D Pose Test\sim_trunk_clothed_vid4_cams_2_4_both.csv"
        # posefile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\3D Pose Test\sim_trunk_clothed_vid5_cams_1_4_both.csv"
        # posefile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\3D Pose Test\sim_trunk_clothed_vid7_cams_2_4_both.csv"

        # posefile = (
        #     r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\3D Pose Test\sim_trunk_limbs_vid7_cams_2_4_both.csv"
        # )

        # LOADING FILE WITH COP VALUES
        # cop_file = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Mat\sim_cop_vid3_each.csv"
        # cop_file = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Mat\sim_cop_vid4_double.csv"
        # cop_file = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Mat\sim_trunk_clothed_cop_vid3_side.csv"
        # cop_file = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Mat\sim_trunk_clothed_cop_vid4_flex.csv"
        # cop_file = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Mat\sim_trunk_clothed_cop_vid5_rot.csv"
        # cop_file = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Mat\sim_trunk_clothed_cop_vid7_side_limbs.csv"

        # cop_file = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Mat\sim_trunk_limbs_cop_vid7_side_static.csv"
        # cop_file = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Mat\sim_trunk_limbs_cop_vid10_all_dynamic.csv"

        posefile = r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\3D Pose Test\sim_trunk_limbs2_vid12_cams_2_4_both.csv"

        cop_file = (
            r"C:\Users\franc\Documents\Infant_Sim_data\passive sim\Mat\sim_trunk_limbs2_cop_vid12_flex_nolimb.csv"
        )

        # self.rate = 60
        # # cop object
        self.COP = processCOP(cop_file, 60)
        self.Fn_tot = self.COP.Rfilt[::2]

        pose = processpose(posefile)

        # pose.adjust_rleg_len()

        pose.IK_init()

        self.R0h = R.from_euler("xzy", np.stack((pose.thet1h[0, :], pose.thet2h[0, :], pose.thet3h[0, :])).T)
        self.R0s = R.from_euler("zy", np.stack((pose.thet1s[0, :], pose.thet2s[0, :])).T)

        self.frames = pose.frames + 1

        self.filter_win = [30, 30, 30, 30]

        self.to_trunk = True
        self.include_limbs = True
        self.fix_Fn = True
        self.plot_dynamics = True

        # self.pose.ID_init()

        ##INITIALIZING PARAMETERS

        # Arm paramters
        # lengths
        uarm_len = (np.mean(pose.get_len(2, 3)) + np.mean(pose.get_len(5, 6))) / 2
        larm_len = (np.mean(pose.get_len(3, 4)) + np.mean(pose.get_len(6, 7))) / 2

        # quit()
        # print(uarm_len, larm_len)

        # radius
        uarm_r = 0.20
        larm_r = 0.15
        # mass
        l_uarm_m = 0.2
        l_larm_m = 0.424 - l_uarm_m - 0.072
        r_uarm_m = 0.22
        r_larm_m = 0.449 - r_uarm_m - 0.075
        # inertia
        l_uarm_I = self.I_limb(uarm_len, l_uarm_m, uarm_r)
        l_larm_I = self.I_limb(larm_len, l_larm_m, larm_r)
        r_uarm_I = self.I_limb(uarm_len, r_uarm_m, uarm_r)
        r_larm_I = self.I_limb(larm_len, r_larm_m, larm_r)

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
        l_uleg_len = np.mean(pose.get_len(11, 12))
        l_lleg_len = np.mean(pose.get_len(12, 13))
        r_uleg_len = np.mean(pose.get_len(8, 9))
        r_lleg_len = np.mean(pose.get_len(9, 10))

        uleg_len = (l_uleg_len + r_uleg_len) / 2
        lleg_len = (l_lleg_len + r_lleg_len) / 2

        l_uleg_len = uleg_len
        l_lleg_len = lleg_len
        r_uleg_len = uleg_len
        r_lleg_len = lleg_len
        # radius
        uleg_r = 0.6
        lleg_r = 0.42
        # mass
        l_uleg_m = 0.5
        l_lleg_m = 0.805 - l_uleg_m
        r_uleg_m = 0.57
        r_lleg_m = 0.823 - r_uleg_m

        # inertia
        l_uleg_I = self.I_limb(l_uleg_len, l_uleg_m, uleg_r)
        l_lleg_I = self.I_limb(l_lleg_len, l_lleg_m, lleg_r)
        r_uleg_I = self.I_limb(r_uleg_len, r_uleg_m, uleg_r)
        r_lleg_I = self.I_limb(r_lleg_len, r_lleg_m, lleg_r)

        # storing left leg parameters
        self.lleg_m = np.array([l_uleg_m, l_lleg_m])
        self.lleg_len = np.array([l_uleg_len, l_lleg_len])
        self.lleg_I = np.matrix([l_uleg_I, l_lleg_I])

        # storing right leg parameters
        self.rleg_m = np.array([r_uleg_m, r_lleg_m])
        self.rleg_len = np.array([r_uleg_len, r_lleg_len])
        self.rleg_I = np.matrix([r_uleg_I, r_lleg_I])

        # print("L Leg", self.lleg_I)
        # print("R Leg", self.rleg_I)

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

        s1 = np.matrix([pose.X[:, 2], pose.Y[:, 2], pose.Z[:, 2]])
        s2 = np.matrix([pose.X[:, 5], pose.Y[:, 5], pose.Z[:, 5]])
        S = (s1 + s2) / 2

        h1 = np.matrix([pose.X[:, 8], pose.Y[:, 8], pose.Z[:, 8]])
        h2 = np.matrix([pose.X[:, 11], pose.Y[:, 11], pose.Z[:, 11]])
        H = (h1 + h2) / 2

        self.m_head = 1.033

        self.L_full = np.linalg.norm(S - H, axis=0)
        self.zerodegm = np.array([0.253, 0.273, 0.150, 0.155])

        # print("Larm Ground", np.min(pose.Z[:, 6]) * 1000)
        # print("Rarm Ground", np.min(pose.Z[:, 3]) * 1000)
        # print("Lleg Ground", np.min(pose.Z[:, 13]) * 1000)
        # print("Rleg Ground", np.min(pose.Z[:, 10]) * 1000)

        self.pose = pose
        # self.plot_limbs()
        # quit()
        # self.view_limb_len()

    def view_limb_len(self):
        pose = self.pose

        plt.subplot(2, 1, 1)
        plt.plot(pose.get_len(8, 9) * 1000)
        plt.plot(pose.get_len(9, 10) * 1000)
        plt.plot(pose.get_len(11, 12) * 1000)
        plt.plot(pose.get_len(12, 13) * 1000)
        plt.legend(["Upper R Leg", "Lower R Leg", "Upper L Leg ", "Lower L Leg "])

        plt.subplot(2, 1, 2)
        plt.plot(pose.get_len(2, 3) * 1000)
        plt.plot(pose.get_len(3, 4) * 1000)
        plt.plot(pose.get_len(5, 6) * 1000)
        plt.plot(pose.get_len(6, 7) * 1000)
        plt.legend(["Upper R Arm", "Lower R Arm", "Upper L Arm ", "Lower L Arm "])
        plt.show()

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

        # for i in [0, 1, 2, 3, 4, 5, 6]:
        for i in range(3):
            Fn = self.Fn[i, j]

            Xcop = np.divide(self.Xcop[i, j], Fn)
            Ycop = np.divide(self.Ycop[i, j], Fn)
            Fn_norm = Fn / self.Fn_tot[j] * 1000
            self.ax.quiver(Xcop, Ycop, -Fn_norm, 0, 0, Fn_norm, color="black")
        # print(Fn)
        # print(Xcop, Ycop, -Fn, Xcop, Ycop, 0)
        Fn_tot = 1000

        self.ax.quiver(Xcalc[j], Ycalc[j], -Fn_tot, 0, 0, Fn_tot, color="red")

        # Fv = self.Flarm[:, j] / self.Fn_tot[j] * 1000
        # self.ax = self.plotF(j, Fv, 5, self.ax)

        Tv = self.Tlarm[:, j] / np.max(self.Tlarm)
        # self.ax = self.plotT(j, Tv, 5, self.ax)

        s = "t= " + str(int(j / 30))
        # s2 = "s= " + str(round(j / 30))

        self.ax.text(0.3, -0.3, 0, "%s" % (s), size=20, zorder=1, color="k")

        # self.ax.set_zlim3d(0, 600)
        # self.ax.set_ylim3d(-100, 10)
        # self.ax.set_xlim3d(0, 160)

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
        ax.arrow(x1, y1, z1, color=[1, 0, 0])
        ax.plot(x2, y2, z2, color="green")
        ax.plot(x3, y3, z3, color=[0, 0, 1])

        return ax

    def plotF(self, i, Fvec, k, ax, scale=1000):
        origin = np.matrix([self.pose.X[i, k], self.pose.Y[i, k], self.pose.Z[i, k]])
        dx = Fvec[0]
        dy = Fvec[1]
        dz = Fvec[2]

        x, y, z = origin[0, 0] * scale, origin[0, 1] * scale, origin[0, 2] * scale

        ax.quiver(x, y, z, dx, 0, 0, colors=[1, 0, 0])
        ax.quiver(x, y, z, 0, dy, 0, colors="green")
        ax.quiver(x, y, z, 0, 0, dz, colors=[0, 0, 1])

        return ax

    def I_trunk(self, L, m):
        lx, ly, lz = L[0], L[1], L[2]

        I1 = (m * (ly**2 + lz**2)) / 12
        I2 = (m * (lx**2 + lz**2)) / 12
        I3 = (m * (lx**2 + ly**2)) / 12

        return np.array([I1, I2, I3])

    def I_limb(self, l, m, r):

        I1 = 1 / 2 * (m * r**2)
        I2 = (1 / 4 * (m * r**2)) + (1 / 12 * (m * l**2))
        # I2 = 1 / 12 * (m * (3 * r**2 + l**2))

        return np.array([I2, I1, I2])

    def plot_ID(self, T, F, title=None):
        fig = plt.figure()

        plt.subplot(2, 1, 1)
        plt.plot(T.T)
        plt.title("Torque")
        plt.legend(["Tx", "Ty", "Tz"])

        plt.subplot(2, 1, 2)
        plt.plot(F.T)
        plt.title("Force")
        plt.legend(["Fx", "Fy", "Fz"])

        if title != None:
            fig.suptitle(title)

        plt.show()

    def plot_ID_LR(self, TL, FL, TR, FR, title=None):
        fig = plt.figure(figsize=(15, 5))

        plt.subplot(2, 2, 1)
        plt.plot(TL.T)
        plt.title("Left Torque")
        plt.legend(["Tx", "Ty", "Tz"])

        plt.subplot(2, 2, 3)
        plt.plot(FL.T)
        plt.title("Left Force")
        plt.legend(["Fx", "Fy", "Fz"])

        plt.subplot(2, 2, 2)
        plt.plot(TR.T)
        plt.title("Right Torque")
        plt.legend(["Tx", "Ty", "Tz"])

        plt.subplot(2, 2, 4)
        plt.plot(FR.T)
        plt.title("Right Force")
        plt.legend(["Fx", "Fy", "Fz"])

        if title != None:
            fig.suptitle(title)

        plt.show()

    def adjust_ID(self, T, F, check):
        T[0, check] = np.ones_like(T[0, check]) * np.mean(T[0, check])
        T[1, check] = np.ones_like(T[1, check]) * np.mean(T[1, check])
        T[2, check] = np.ones_like(T[2, check]) * np.mean(T[2, check])

        F[0, check] = np.ones_like(F[0, check]) * np.mean(F[0, check])
        F[1, check] = np.ones_like(F[1, check]) * np.mean(F[1, check])
        F[2, check] = np.ones_like(F[2, check]) * np.mean(F[2, check])

        ncheck = np.logical_not(check)

        T[0, ncheck] = np.ones_like(T[0, ncheck]) * np.mean(T[0, ncheck])
        T[1, ncheck] = np.ones_like(T[1, ncheck]) * np.mean(T[1, ncheck])
        T[2, ncheck] = np.ones_like(T[2, ncheck]) * np.mean(T[2, ncheck])

        F[0, ncheck] = np.ones_like(F[0, ncheck]) * np.mean(F[0, ncheck])
        F[1, ncheck] = np.ones_like(F[1, ncheck]) * np.mean(F[1, ncheck])
        F[2, ncheck] = np.ones_like(F[2, ncheck]) * np.mean(F[2, ncheck])

        return T, F

    def plot_limbs(self):
        pose = self.pose

        plt.figure(figsize=(10, 9))

        plt.subplot(2, 2, 2)
        plt.plot(np.rad2deg(pose.thet1_ra.T))
        plt.plot(np.rad2deg(pose.thet2_ra.T))
        plt.plot(np.rad2deg(pose.thet3_ra.T))
        plt.plot(np.rad2deg(pose.thet4_ra.T))
        plt.legend(["thet1", "thet2", "thet3", "thet4"])
        plt.title("Right Arm")
        plt.xlabel("Time (s)")
        plt.ylabel("Degrees")

        plt.subplot(2, 2, 1)
        plt.plot(np.rad2deg(pose.thet1_la.T))
        plt.plot(np.rad2deg(pose.thet2_la.T))
        plt.plot(np.rad2deg(pose.thet3_la.T))
        plt.plot(np.rad2deg(pose.thet4_la.T))
        plt.legend(["thet1", "thet2", "thet3", "thet4"])
        plt.title("Left Arm")
        plt.xlabel("Time (s)")
        plt.ylabel("Degrees")

        plt.subplot(2, 2, 4)
        plt.plot(np.rad2deg(pose.thet1_rl.T))
        plt.plot(np.rad2deg(pose.thet2_rl.T))
        plt.plot(np.rad2deg(pose.thet3_rl.T))
        plt.plot(np.rad2deg(pose.thet4_rl.T))
        plt.legend(["thet1", "thet2", "thet3", "thet4"])
        plt.title("Right Leg")
        plt.xlabel("Time (s)")
        plt.ylabel("Degrees")

        plt.subplot(2, 2, 3)
        plt.plot(np.rad2deg(pose.thet1_ll.T))
        plt.plot(np.rad2deg(pose.thet2_ll.T))
        plt.plot(np.rad2deg(pose.thet3_ll.T))
        plt.plot(np.rad2deg(pose.thet4_ll.T))
        plt.legend(["thet1", "thet2", "thet3", "thet4"])
        plt.title("Left Leg")
        plt.xlabel("Time (s)")
        plt.ylabel("Degrees")

        plt.show()

    def step_Theta(self, T, F, Theta, t1, t2, filt_window=1, left=False):
        thet3 = scipy.ndimage.median_filter(np.rad2deg(Theta[2, :]), [filt_window])

        if left:
            thet3 = -thet3

        for i in range(3):
            # Theta[i, thet3 < t1] = np.mean(Theta[i, thet3 < t1]) * np.ones_like(Theta[i, thet3 < t1])
            # check = np.logical_and(thet3 > t1, thet3 < t2)
            # Theta[i, check] = np.mean(Theta[i, check])
            # Theta[i, thet3 > t2] = np.mean(Theta[i, thet3 > t2]) * np.ones_like(Theta[i, thet3 > t2])

            check1 = thet3 < t1
            check2 = np.logical_and(thet3 > t1, thet3 < t2)
            check3 = thet3 > t2

            T[i, check1] = np.ones_like(T[i, check1]) * np.mean(T[i, check1])
            F[i, check1] = np.ones_like(F[i, check1]) * np.mean(F[i, check1])

            T[i, check2] = np.ones_like(T[i, check2]) * np.mean(T[i, check2])
            F[i, check2] = np.ones_like(F[i, check2]) * np.mean(F[i, check2])

            T[i, check3] = np.ones_like(T[i, check3]) * np.mean(T[i, check3])
            F[i, check3] = np.ones_like(F[i, check3]) * np.mean(F[i, check3])

        # plt.plot(np.rad2deg(Theta[0, :]))
        # plt.plot(np.rad2deg(Theta[1, :]))
        # plt.plot(np.rad2deg(Theta[2, :]))
        # plt.plot(np.rad2deg(Theta[3, :]))
        # plt.legend(["thet1", "thet2", "thet3", "thet4"])
        # plt.xlabel("Time (s)")
        # plt.ylabel("Degrees")

        # plt.show()
        # quit()

        return T, F

    def filter_dynamics(self, T, F, win):
        T, F = scipy.ndimage.median_filter(T, [1, win]), scipy.ndimage.median_filter(F, [1, win])

        return T, F

    def get_theta(self, k):
        pose = self.pose

        if k == 2 or k == 5:
            thet1, thet2, thet3, thet4 = pose.arm_ang(k)
        elif k == 8 or k == 11:
            thet1, thet2, thet3, thet4 = pose.leg_ang(k)

        Theta = np.array([thet1, thet2, thet3, thet4])[:, 0, :]

        return Theta

    def COP_upper(self):
        g = 9.81
        pose = self.pose

        # quit()

        Xlow, Ylow, FnLow, Tx, Ty = self.COP_lower()

        # Upper trunk parameters
        l = self.utrunk_l
        w = self.utrunk_w
        M = self.utrunk_m
        h = self.utrunk_h

        Iup = self.I_trunk(L=np.array([w, l, h]), m=M)
        Lup = np.mean(pose.get_len(2, 5)) / 2

        Tqup, w_up, w_d_up, Vls_d, Vrs_d = pose.inv_dynamics_upper(Lup, Iup)

        # self.plot_ID(Tqup, Tqup)
        # quit()

        # right arm dynamics
        Theta_rarm = self.get_theta(2)
        Trarm, Frarm, e1r, e2r = pose.inv_dynamics(
            Theta_rarm,
            self.rarm_len,
            self.rarm_m,
            self.rarm_I,
            W0=w_up,
            W0_d=w_d_up,
            V0_d=Vrs_d,
            to_upper=self.to_trunk,
        )
        # Trarm, Frarm = self.step_Theta(Trarm, Frarm, Theta_rarm, t1=-37.5, t2=0, filt_window=1, left=False)
        Trarm, Frarm = self.filter_dynamics(Trarm, Frarm, self.filter_win[0])

        # rarm_check = np.ravel((np.rad2deg(pose.thet3_ra) < -45))
        # Trarm, Frarm = self.adjust_ID(Trarm, Frarm, rarm_check)

        # rarm_check = np.ravel((np.rad2deg(pose.thet3_ra) < -45))
        # rarm_check = np.ravel(scipy.ndimage.median_filter(np.rad2deg(pose.thet3_ra), [1, 100]) < -20)
        # vid 7
        # rarm_check = np.ravel((np.rad2deg(pose.thet3_ra) < -35))
        # vid 8
        # rarm_check = np.ravel((np.rad2deg(pose.thet3_ra) < -15))

        # print(np.mean(Trarm[:, rarm_check], axis=1
        # Trarm, Frarm = self.adjust_ID(Trarm, Frarm, rarm_check)
        # self.plot_ID(Trarm, Frarm, title="Rarm Dynamics")
        # quit()

        # Wrinting dynamic terms for right arm
        T1x = Trarm[0, 0 : self.frames]
        T1y = Trarm[1, 0 : self.frames]

        F1x = Frarm[0, 0 : self.frames]
        F1y = Frarm[1, 0 : self.frames]
        F1z = Frarm[2, 0 : self.frames]

        # left arm dynamics
        # Theta_larm = self.get_theta(5)
        Theta_larm = self.get_theta(5)
        Tlarm, Flarm, e1l, e2l = pose.inv_dynamics(
            Theta_larm,
            self.larm_len,
            self.larm_m,
            self.larm_I,
            W0=w_up,
            W0_d=w_d_up,
            V0_d=Vls_d,
            to_upper=self.to_trunk,
        )
        # Tlarm, Flarm = self.step_Theta(Tlarm, Flarm, Theta_larm, t1=-50, t2=0, filt_window=1, left=True)
        Tlarm, Flarm = self.filter_dynamics(Tlarm, Flarm, self.filter_win[1])

        # larm_check = np.ravel(np.rad2deg(pose.thet3_la) > 50)
        # Tlarm, Flarm = self.adjust_ID(Tlarm, Flarm, larm_check)

        # larm_check = np.ravel(np.rad2deg(pose.thet3_la) > 50)
        # larm_check = np.ravel(np.rad2deg(pose.thet3_la) > 40)

        # Tlarm, Flarm = self.adjust_ID(Tlarm, Flarm, larm_check)
        self.Tlarm, self.Flarm = Tlarm, Flarm
        # self.plot_ID(Tlarm, Flarm, title="Larm Dynamics")

        # Wrinting dynamic terms for left arm
        T2x = Tlarm[0, 0 : self.frames]
        T2y = Tlarm[1, 0 : self.frames]

        F2x = Flarm[0, 0 : self.frames]
        F2y = Flarm[1, 0 : self.frames]
        F2z = Flarm[2, 0 : self.frames]

        self.plot_ID_LR(Tlarm, Flarm, Trarm, Frarm, title="Arm Dynamics")

        Txx = (T1x + T2x) + Tx - Tqup[0, 0 : self.frames]
        Tyy = (T1y + T2y) + Ty - Tqup[1, 0 : self.frames]

        # plt.plot(Txx)
        # plt.plot(Tyy)
        # plt.legend(["Txx", "Tyy"])
        # plt.show()

        mg = g * M
        F12_x = F1x + F2x
        F12_y = F1y + F2y
        F12_z = F1z + F2z

        # print(np.shape(Tyy),np.shape(F1z))

        dx = w / 2
        dy = l / 2
        dz = h / 2

        # X_calc = np.divide((Tyy - (F12_x * dz) + ((F1z - F2z) * dx)), (mg - F12_z))
        # Y_calc = np.divide((Txx - (F12_y * dz) + (F12_z * dy)), (F12_z - mg))

        # X_calc = ((T1y + T2y + Ty) - ((F1x + F2x) * h * 0.5) + ((F1z - F2z) * w * 0.5)) / ((g * M) - (F1z + F2z))
        # Y_calc = ((T1x + T2x + Tx) + ((F1z + F2z) * l * 0.5) - ((F1y + F2y) * h * 0.5)) / ((F1z + F2z) - (g * M))

        # print(np.shape(Txx), np.shape(F1z), self.frames)

        X_calc = np.divide((Tyy + ((F1z - F2z) * dx) + (F12_x * dz)), (mg - F12_z))
        Y_calc = np.divide((Txx - (F12_y * dz) + (F12_z * dy)), (F12_z - mg))

        # quit()

        # plt.plot((T1y + T2y + Ty))
        # plt.plot(((F1z - F2z) * w * 0.5))
        # plt.legend(["Ty", "Fz"])
        # plt.show()

        # X_calc = ((T1y + T2y + Tt) + ((F1z - F2z) * w * 0.5)) / (-F1z - F2z + (g * M))
        # Y_calc = ((T1x - T2x) + ((F1z + F2z) * l * 0.5)) / (F1z + F2z - (g * M))

        mid_proj = self.R0s.apply(np.array([0, -75, 0]) / 1000)

        COM = (pose.mid_shoulder.T + mid_proj).T

        # X_calc = COM[0, 0 : self.frames]
        # Y_calc = COM[1, 0 : self.frames]

        # COM_X = (pose.mid_shoulder.T[:, 0] + mid_proj)[0 : self.frames, 0]
        # COM_Y = (pose.mid_shoulder.T[:, 1] + mid_proj)[0 : self.frames, 1]

        X_calc = COM[0, 0 : self.frames] + (X_calc * int(self.include_limbs))
        # Y_calc = COM_Y - dy + Y_calc
        Y_calc = COM[1, 0 : self.frames] + (Y_calc * int(self.include_limbs))

        Fn_up = (g * M * np.ones_like(X_calc)) - F1z - F2z

        Xup = np.multiply(X_calc, Fn_up)
        Yup = np.multiply(Y_calc, Fn_up)

        return Xup, Yup, Fn_up, Xlow, Ylow, FnLow

    def COP_lower(self):
        pose = self.pose
        g = 9.81

        # lower trunk parameters
        l = self.ltrunk_l
        w = self.ltrunk_w
        M = self.ltrunk_m
        h = self.ltrunk_h

        # Ilow = self.I_trunk(L=np.array([0.180, 0.110, 0.080]), m=M)
        Ilow = self.I_trunk(L=np.array([w, l, h]), m=M)
        Llow = np.mean(pose.get_len(8, 11)) / 2

        Tqlow, w_low, w_d_low, Vlh_d, Vrh_d = pose.inv_dynamics_lower(Llow, Ilow, to_upper=True)

        # self.plot_ID(w_low, w_d_low)
        # self.plot_ID(Vlh_d, Vrh_d)
        # self.plot_ID(Tqlow, Vrh_d)

        # right leg dynamics
        Theta_rleg = self.get_theta(8)

        Trleg, Frleg, e1r, e2r = pose.inv_dynamics(
            Theta_rleg,
            self.rleg_len,
            self.rleg_m,
            self.rleg_I,
            W0=w_low,
            W0_d=w_d_low,
            V0_d=Vrh_d,
            to_lower=self.to_trunk,
        )

        # Trleg, Frleg = self.step_Theta(Trleg, Frleg, Theta_rleg, t1=25, t2=100, filt_window=45, left=False)
        Trleg, Frleg = self.filter_dynamics(Trleg, Frleg, self.filter_win[2])

        # rleg_check = np.ravel(np.rad2deg(pose.thet3_rl) < 50)
        # Trleg, Frleg = self.adjust_ID(Trleg, Frleg, rleg_check)

        # zeroing force and torque when leg is at rest
        # rleg_check = np.ravel(np.rad2deg(pose.thet3_rl) < 50)

        # vid 7
        # rleg_check = np.ravel(scipy.ndimage.median_filter(np.rad2deg(pose.thet3_rl), [1, 100]) < 110)
        # vid 8
        # rleg_check = np.ravel(scipy.ndimage.median_filter(np.rad2deg(pose.thet3_rl), [1, 140]) < 125)

        # Trleg, Frleg = self.adjust_ID(Trleg, Frleg, rleg_check)
        # self.plot_ID(Trleg, Frleg, title="Rleg Dynamics")

        # right leg dynamic terms
        T3x, T3y = Trleg[0, :], Trleg[1, :]
        F3x, F3y, F3z = Frleg[0, :], Frleg[1, :], Frleg[2, :]

        Theta_lleg = self.get_theta(11)
        # Theta_lleg = self.step_Theta(self.get_theta(11), t1=75, t2=125, filt_window=1, left=True)
        Tlleg, Flleg, e1l, e2l = pose.inv_dynamics(
            Theta_lleg,
            self.lleg_len,
            self.lleg_m,
            self.lleg_I,
            W0=w_low,
            W0_d=w_d_low,
            V0_d=Vlh_d,
            to_lower=self.to_trunk,
        )

        # Tlleg, Flleg = self.step_Theta(Tlleg, Flleg, Theta_lleg, t1=75, t2=125, filt_window=1, left=True)
        Tlleg, Flleg = self.filter_dynamics(Tlleg, Flleg, self.filter_win[3])

        # lleg_check = np.ravel(np.rad2deg(pose.thet3_ll) > -50)
        # Tlleg, Flleg = self.adjust_ID(Tlleg, Flleg, lleg_check)

        # Adjsuting/smoothing dy
        # lleg_check = np.ravel(np.rad2deg(pose.thet3_ll) > -50)
        # lleg_check = np.ravel(scipy.ndimage.median_filter(np.rad2deg(pose.thet3_ll), [1, 55]) > -100)
        # Tlleg, Flleg = self.adjust_ID(Tlleg, Flleg, lleg_check)
        # self.plot_ID(Tlleg, Flleg, title="Lleg Dynamics")

        # left leg dynamic terms
        T4x, T4y = Tlleg[0, :], Tlleg[1, :]
        F4x, F4y, F4z = Flleg[0, :], Flleg[1, :], Flleg[2, :]

        # win = 15
        # Tlleg = scipy.ndimage.median_filter(Tlleg, win)
        # Flleg = scipy.ndimage.median_filter(Flleg, win)
        # Trleg = scipy.ndimage.median_filter(Trleg, win)
        # Frleg = scipy.ndimage.median_filter(Frleg, win)

        self.plot_ID_LR(Tlleg, Flleg, Trleg, Frleg, title="Leg Dynamics")
        # quit()

        F34_x = F3x + F4x
        F34_y = F3y + F4y
        F34_z = F3z + F4z

        T34_x = T3x + T4x
        T34_y = T3y + T4y

        dx = w / 2
        dy = l / 4
        dz = h / 2

        tx = T34_x + Tqlow[0, :] - (F34_z * dy) - (F34_y * dz)
        ty = T34_y + Tqlow[1, :] + ((F3z - F4z) * dx) + (F34_x * dz)
        tz = np.zeros_like(tx)

        Tlow = np.stack((tx, ty, tz))
        Rhs = self.R0h.inv() * self.R0s
        Tlow_cor = Rhs.apply(Tlow.T).T

        # print(np.shape(Tlow), np.shape(Tlow_cor))
        Tx = Tlow_cor[0, :]
        Ty = Tlow_cor[1, :]

        # plt.plot(np.diff(np.rad2deg(pose.thet3_rl)).T)
        # # plt.plot(Ty)
        # plt.show()
        # quit()

        # Fn = (g * M * np.ones_like(F3z)) - F3z - F4z

        M_low = M * np.ones_like(F3z)

        M_low_d = self.adjust_Fn(M_low)

        Fn = (M_low_d[0 : self.frames] * g) - F3z[0 : self.frames] - F4z[0 : self.frames]

        # plt.plot((M * np.ones_like(F3z)))
        # plt.plot(M_low_d)
        # plt.legend(["b4", "after"])
        # plt.show()
        # quit()

        rtrunk = 91.39

        mid_proj = self.R0h.apply(np.array([0, 0, (rtrunk - 80)]) / 1000)

        COM = (pose.mid_hip.T + mid_proj).T

        X_calc = COM[0, 0 : self.frames]
        Y_calc = COM[1, 0 : self.frames]

        # X_calc = np.ones_like(COM_X) * np.mean((pose.X[:, 8] + pose.X[:, 11]) * 0.5)
        # Y_calc = np.ones_like(COM_Y) * np.mean((pose.Y[:, 8] + pose.Y[:, 11]) * 0.5)

        # plt.plot(COM_X)
        # plt.plot(COM_Y)
        # plt.show()

        # self.rleg_check, self.lleg_check = rleg_check, lleg_check

        return (
            np.multiply(X_calc, Fn),
            np.multiply(Y_calc, Fn),
            np.reshape(Fn, [1, self.frames]),
            Tx[0 : self.frames],
            Ty[0 : self.frames],
        )

    def COP_trunk(self):
        pose = self.pose
        g = 9.81

        rtrunk = 91.39

        # lower trunk parameters
        w1, w2 = self.utrunk_w, self.ltrunk_w
        l1, l2 = self.utrunk_l, self.ltrunk_l
        h1, h2 = self.utrunk_h, self.ltrunk_h
        M1, M2 = self.utrunk_m, self.ltrunk_m

        dx_up, dy_up, dz_up = w1 / 2, l1 / 2, h1 / 2
        dx_low, dy_low, dz_low = w2 / 2, l2 / 4, h2 / 2

        Iup = self.I_trunk(L=np.array([w1, l1, h1]), m=M1)
        Ilow = self.I_trunk(L=np.array([w2, l2, h2]), m=M2)
        Lup = np.mean(pose.get_len(2, 5)) / 2
        Llow = np.mean(pose.get_len(8, 11)) / 2

        mg_up = g * M1
        mg_low = g * M2

        Tqlow, w_low, w_d_low, Vlh_d, Vrh_d = pose.inv_dynamics_lower(Llow, Ilow, to_upper=True)
        Tqup, w_up, w_d_up, Vls_d, Vrs_d = pose.inv_dynamics_upper(Lup, Iup)

        # Right Arm dynamics
        Theta_rarm = self.get_theta(2)
        Trarm, Frarm, e1r, e2r = pose.inv_dynamics(
            Theta_rarm,
            self.rarm_len,
            self.rarm_m,
            self.rarm_I,
            W0=w_up,
            W0_d=w_d_up,
            V0_d=Vrs_d,
            to_upper=self.to_trunk,
        )

        # Left Arm dynamics
        Theta_larm = self.get_theta(5)
        Tlarm, Flarm, e1l, e2l = pose.inv_dynamics(
            Theta_larm,
            self.larm_len,
            self.larm_m,
            self.larm_I,
            W0=w_up,
            W0_d=w_d_up,
            V0_d=Vls_d,
            to_upper=self.to_trunk,
        )

        # Right Leg Dynamics
        Theta_rleg = self.get_theta(8)
        Trleg, Frleg, e1r, e2r = pose.inv_dynamics(
            Theta_rleg,
            self.rleg_len,
            self.rleg_m,
            self.rleg_I,
            W0=w_low,
            W0_d=w_d_low,
            V0_d=Vrh_d,
            to_lower=self.to_trunk,
        )

        # Left Leg Dynamics
        Theta_lleg = self.get_theta(11)
        Tlleg, Flleg, e1l, e2l = pose.inv_dynamics(
            Theta_lleg,
            self.lleg_len,
            self.lleg_m,
            self.lleg_I,
            W0=w_low,
            W0_d=w_d_low,
            V0_d=Vlh_d,
            to_lower=self.to_trunk,
        )

        # Filtering Dynamics
        Trarm, Frarm = self.filter_dynamics(Trarm, Frarm, self.filter_win[0])
        Tlarm, Flarm = self.filter_dynamics(Tlarm, Flarm, self.filter_win[1])
        Trleg, Frleg = self.filter_dynamics(Trleg, Frleg, self.filter_win[2])
        Tlleg, Flleg = self.filter_dynamics(Tlleg, Flleg, self.filter_win[3])

        # Wrinting dynamic terms for right arm
        T1x, T1y = Trarm[0, :], Trarm[1, :]
        F1x, F1y, F1z = Frarm[0, :], Frarm[1, :], Frarm[2, :]

        # Wrinting dynamic terms for left arm
        T2x, T2y = Tlarm[0, :], Tlarm[1, :]
        F2x, F2y, F2z = Flarm[0, :], Flarm[1, :], Flarm[2, :]

        # Wrinting dynamic terms for right leg
        T3x, T3y = Trleg[0, :], Trleg[1, :]
        F3x, F3y, F3z = Frleg[0, :], Frleg[1, :], Frleg[2, :]

        # Wrinting dynamic terms for left leg
        T4x, T4y = Tlleg[0, :], Tlleg[1, :]
        F4x, F4y, F4z = Flleg[0, :], Flleg[1, :], Flleg[2, :]

        # plt.plot(F1z)
        # plt.plot(F2z)
        # plt.plot(F3z)
        # plt.plot(F4z)
        # plt.legend(["rarm", "larm", "rleg", "lleg"])
        # plt.show()
        # quit()

        # Getting Fn for upper and lower trunk

        M_up = M1 * np.ones_like(F1z)
        M_low = M2 * np.ones_like(F3z)

        if self.fix_Fn:
            Fn_limbs = F1z + F2z + F3z + F4z
            M_low = self.adjust_Fn(M_low, Fn_limbs)

        Fn_up = (M_up[0 : self.frames] * g) - F1z[0 : self.frames] - F2z[0 : self.frames]
        Fn_low = (M_low[0 : self.frames] * g) - F3z[0 : self.frames] - F4z[0 : self.frames]
        # Fn_low = M_low[0 : self.frames] * g

        # Precombining Terms
        F12_x, F12_y, F12_z = F1x + F2x, F1y + F2y, F1z + F2z
        F34_x, F34_y, F34_z = F3x + F4x, F3y + F4y, F3z + F4z
        T12_x, T12_y = T1x + T2x, T1y + T2y
        T34_x, T34_y = T3x + T4x, T3y + T4y

        ## Lower Trunk COP Calculation
        tx = T34_x + Tqlow[0, :] - (F34_z * dy_low) - (F34_y * dz_low)
        ty = T34_y + Tqlow[1, :] + ((F3z - F4z) * dx_low) + (F34_x * dz_low)
        tz = np.zeros_like(tx)

        # Converting Lower Trunk Torque into Upper Trunk ref. frame
        Tlow = np.stack((tx, ty, tz))
        Rhs = self.R0h.inv() * self.R0s
        Tlow_cor = Rhs.apply(Tlow.T).T

        Tx_low = Tlow_cor[0, :]
        Ty_low = Tlow_cor[1, :]

        # Getting Lower Trunk torque
        Tx_up = (T12_x) + Tx_low - Tqup[0, :]
        Ty_up = (T12_y) + Ty_low - Tqup[1, :]

        # calculating COP change from upper limbs on upper trunk
        X_calc_up = np.divide((Ty_up + ((F1z - F2z) * dx_up) + (F12_x * dz_up)), (mg_up - F12_z))
        Y_calc_up = np.divide((Tx_up - (F12_y * dz_up) + (F12_z * dy_up)), (F12_z - mg_up))

        # Calculating Projection of COM offest
        mid_proj_up = self.R0s.apply(np.array([0, -75, 0]) / 1000)
        mid_proj_low = self.R0h.apply(np.array([0, 0, (rtrunk - 80)]) / 1000)

        # Getting Full COM location by adding offset
        COM_up = (pose.mid_shoulder.T + mid_proj_up).T
        COM_low = (pose.mid_hip.T + mid_proj_low).T

        # Calculating unweighted upper trunk COP
        X_calc_up = (COM_up[0, :] + (X_calc_up * int(self.include_limbs)))[:, 0 : self.frames]
        Y_calc_up = (COM_up[1, :] + (Y_calc_up * int(self.include_limbs)))[:, 0 : self.frames]

        COP_Xup = np.multiply(X_calc_up, Fn_up)
        COP_Yup = np.multiply(Y_calc_up, Fn_up)

        COP_Xlow = np.multiply(COM_low[0, 0 : self.frames], Fn_low)
        COP_Ylow = np.multiply(COM_low[1, 0 : self.frames], Fn_low)

        if self.plot_dynamics:
            self.plot_ID_LR(Tlarm, Flarm, Trarm, Frarm, title="Arm Dynamics")
            self.plot_ID_LR(Tlleg, Flleg, Trleg, Frleg, title="Leg Dynamics")

        return COP_Xup, COP_Yup, Fn_up, COP_Xlow, COP_Ylow, Fn_low

    def adjust_Fn(self, M_low, Fn_limbs):
        g = 9.81

        # M = np.array(
        #     [
        #         self.m_head,
        #         self.utrunk_m,
        #         np.sum(self.rarm_m),
        #         np.sum(self.larm_m),
        #         np.sum(self.rleg_m),
        #         np.sum(self.lleg_m),
        #     ]
        # )

        M = np.array(
            [
                self.m_head,
                self.utrunk_m,
            ]
        )

        Fn = self.Fn_tot

        # m_head, m_up, m_low = M[0], M[1], M[2]
        m_mat = 1.325

        # print(np.shape(M))

        M_low_d = Fn - (np.sum(M) + m_mat)

        # if Fn_limbs != None:

        n = min(np.shape(M_low)[0], np.shape(M_low_d)[0])
        self.frames = n

        # print(np.sum(self.rarm_m) + np.sum(self.larm_m) + np.sum(self.rleg_m) + np.sum(self.lleg_m))
        # print(np.mean(-(Fn_limbs[0:n] / 9.8)))
        # quit()

        M_low_d = (M_low_d[0:n] * g) + Fn_limbs[0:n]
        # M_low_d = M_low_d[0:n] - (np.sum(self.rarm_m) + np.sum(self.larm_m) + np.sum(self.rleg_m) + np.sum(self.lleg_m))

        # plt.plot(M_low[0:n] * g)
        # plt.plot(M_low_d[0:n])
        # plt.plot(-(Fn_limbs[0:n]))
        # plt.legend(["mg", "comp_nolimbs", "limbs"])
        # plt.show()
        # quit()
        return M_low_d / g

    def calc_COP(self):
        # X_up, Y_up, Fn_up, X_low, Y_low, Fn_low = self.COP_upper()
        X_up, Y_up, Fn_up, X_low, Y_low, Fn_low = self.COP_trunk()

        Fn_head = np.ones_like(Fn_up) * self.m_head * 9.81
        X_head = np.multiply(self.pose.X[0 : self.frames, 0], X_up)
        Y_head = np.multiply(self.pose.Y[0 : self.frames, 0], Y_up)

        ##UPPER/LOWER TRUNK
        Fn_tot = Fn_up + Fn_low + Fn_head

        self.Xcalc = (X_low + X_up + X_head) / Fn_tot * 1000
        self.Ycalc = (Y_low + Y_up + Y_head) / Fn_tot * 1000

        # print(np.shape(X_head), np.shape(X_up), np.shape(X_low))
        # print(np.shape(Fn_head), np.shape(Fn_up), np.shape(Fn_low))

        self.Fn = np.array([Fn_low, Fn_up, Fn_head])
        self.Xcop = np.array([X_low, X_up, X_head]) * 1000
        self.Ycop = np.array([Y_low, Y_up, Y_head]) * 1000

        self.Fn_tot = Fn_tot

        # plt.plot(X_up)
        # plt.plot(X_low)
        # plt.plot(X_head)
        # plt.plot(self.Xcalc / 1000)

        # plt.plot(Fn_up)
        # plt.plot(Fn_low)
        # plt.plot(Fn_head)
        # plt.plot(Fn_tot)

        # plt.legend(["Upper", "Lower", "Head", "Tot"])
        # plt.grid()
        # plt.show()

    def compare_COP(self):
        Xcalc = self.Xcalc.T
        Ycalc = self.Ycalc.T

        # COP_gc = processCOP(file, 60)

        Xreal = self.COP.Xfilt[::2]
        Yreal = self.COP.Yfilt[::2]

        # cc = compareCOP(COP_X, COP_Y, X_gc, Y_gc)

        # fig, ax = plt.subplots(2, 1)
        ##Left Arm
        # start = delay
        # stop = 1350

        ##Righ Arm
        # start = 1350
        # stop = 2600

        ##Left Leg
        # start = 2600
        # stop = 3900

        # ##Right Leg
        # start = 3800
        # stop = -1

        ##side limbs static
        start = 200
        stop = -1

        Xreal, Yreal = Xreal[start:stop], Yreal[start:stop]
        Xcalc, Ycalc = Xcalc[start:stop], Ycalc[start:stop]

        win = 2

        Xcalc = scipy.ndimage.median_filter(Xcalc, win)
        Ycalc = scipy.ndimage.median_filter(Ycalc, win)

        cc = compareCOP(Xcalc, Ycalc, Xreal, Yreal, cam=1)
        cc.comp_XY()
        # cc.comp_ellipse()
        # cc.plot_cop_anim()

        # print(cc.metrics())
        print(cc.diff_metric())


test = sim_COP()
test.calc_COP()
# test.calc_COP_full()
print("calculation done")
# test.plot_cop_anim()
test.compare_COP()
