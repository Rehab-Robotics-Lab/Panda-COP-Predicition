# from Inv_dynamics import inv_dynamics
from ProcessCOP import processCOP
from CompareCOP import compareCOP
from Infant_Paramters import infant_params
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import scipy
from scipy import stats
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from ProcessPose_3D import processpose


class calculate_COP:
    def __init__(
        self,
        posefile,
        cop_file,
        head="face",
    ):
        # # cop object
        self.COP = processCOP(cop_file, 60, fcut=1)
        self.Fn_mat = self.COP.Rfilt[::2]

        pose = processpose(posefile)

        # pose.adjust_rleg_len()
        self.head = head

        pose.IK_init()

        self.R0h = R.from_euler("XZY", np.stack((pose.thet1h[:], pose.thet2h[:], pose.thet3h[:])).T)
        self.R0s = R.from_euler("ZY", np.stack((pose.thet1s[0, :], pose.thet2s[0, :])).T)

        if head == "face":
            self.R0f = R.from_euler("XZY", np.stack((pose.thet1f[:], pose.thet2f[:], pose.thet3f[:])).T)
            self.deg_face = np.deg2rad(30)
        elif head == "ears":
            self.R0f = R.from_euler("XZY", np.stack((pose.thet1e[0:], pose.thet2e[0:], pose.thet3e[0:])).T)
            self.deg_face = np.deg2rad(-30)

        self.frames = pose.frames + 1

        self.filter_win = np.array([1, 1, 1, 1]) * 10

        self.to_trunk = True
        self.to_world = True

        self.include_limbs = True
        self.include_head = True

        self.plot_dynamics = False

        # self.pose.ID_init()

        ##INITIALIZING PARAMETERS

        self.head_h_ratio_chest = 1.5
        self.r_head_ratio = 0.5

        self.r_trunk_ratio = 1
        self.trunk_h_ratio = 0.75
        self.deg_trunk = np.deg2rad(0)
        self.utrunk_com_ratio = 0.35

        self.a_head = 0.1
        self.b_head = 0.1
        self.c_head = 0.1
        self.z_off_head = 0.1

        self.center_low = 0
        self.center_head = 0
        self.center_up_y = 0.05
        self.center_up_z = 0

        # age = param.guess_age_params(mass, forearm)
        # age = param.age_from_mass(mass)
        # age = param.age_from_length(forearm, "low arm")

        param = infant_params()
        age = 3 * 4
        # print(age)

        # Ma = param.get_mass(age)
        # Ma = Ma / Ma.loc["total"]
        # print(Ma)\
        self.tot_mass = 10
        m = np.array(
            [0.243552 + 0.012891, 0.187610, 0.233627, 0.021043, 0.021359 + 0.007581, 0.065917, 0.033411 + 0.011849]
        )
        # m = np.array([0.2, 0.18, 0.35, 0.021043, 0.021359 + 0.007581, 0.065917, 0.033411 + 0.011849])
        # m = np.array([0.2, 0.15, 0.45, 0.021043, 0.021359 + 0.007581, 0.065917, 0.033411 + 0.011849])
        self.m_dist = m
        Ma = m * self.tot_mass

        uarm_m = Ma[3]
        larm_m = Ma[4]
        uleg_m = Ma[5]
        lleg_m = Ma[6]

        # storing arm parameters
        self.arm_m = np.array([uarm_m, larm_m])
        self.leg_m = np.array([uleg_m, lleg_m])
        self.utrunk_m = Ma[1]
        self.ltrunk_m = Ma[2]
        self.m_head = Ma[0]

        Ra = param.get_radius(age)
        In = param.get_inertia(age)
        Le = param.get_length(age)

        uleg_r = 0.6
        lleg_r = 0.42
        uarm_r = 0.20
        larm_r = 0.15

        # Arm paramters
        # lengths
        uarm_len = (np.mean(pose.get_len(2, 3)) + np.mean(pose.get_len(5, 6))) / 2
        larm_len = (np.mean(pose.get_len(3, 4)) + np.mean(pose.get_len(6, 7))) / 2

        # quit()
        # print(uarm_len, larm_len)

        # inertia
        uarm_I = np.array([In.loc["upp arm", "Ix"], In.loc["upp arm", "Iy"], In.loc["upp arm", "Iz"]])
        larm_I = np.array([In.loc["low arm", "Ix"], In.loc["low arm", "Iy"], In.loc["low arm", "Iz"]])

        # uarm_I = self.I_limb(uarm_len, uarm_m, uarm_r)
        # larm_I = self.I_limb(larm_len, larm_m, larm_r)

        # storing arm parameters
        self.arm_len = np.array([uarm_len, larm_len])
        self.arm_I = np.matrix([uarm_I, larm_I])

        # Leg paramters
        # lengths
        uleg_len = (np.mean(pose.get_len(11, 12)) + np.mean(pose.get_len(8, 9))) / 2
        lleg_len = (np.mean(pose.get_len(12, 13)) + np.mean(pose.get_len(9, 10))) / 2

        # inertia
        uleg_I = np.array([In.loc["upp leg", "Ix"], In.loc["upp leg", "Iy"], In.loc["upp leg", "Iz"]])
        lleg_I = np.array([In.loc["low leg", "Ix"], In.loc["low leg", "Iy"], In.loc["low leg", "Iz"]])

        # uleg_I = self.I_limb(uleg_len, uleg_m, uleg_r)
        # lleg_I = self.I_limb(lleg_len, lleg_m, lleg_r)

        # storing left leg parameters
        self.leg_len = np.array([uleg_len, lleg_len])
        self.leg_I = np.matrix([uleg_I, lleg_I])

        # Upper Trunk parameters
        self.utrunk_l = Le.loc["upp trunk"][0]
        self.utrunk_w = np.mean(pose.get_len(2, 5))
        # self.utrunk_w = 0.18
        self.utrunk_h = Ra.loc["upp trunk"][0]

        # Lower Trunk parameters
        self.ltrunk_l = Le.loc["low trunk"][0]
        self.ltrunk_w = np.mean(pose.get_len(8, 11))
        # self.ltrunk_w = 0.18
        self.ltrunk_h = Ra.loc["low trunk"][0]

        self.Iup = np.array([In.loc["upp trunk", "Ix"], In.loc["upp trunk", "Iy"], In.loc["upp trunk", "Iz"]])
        self.Ilow = np.array([In.loc["low trunk", "Ix"], In.loc["low trunk", "Iy"], In.loc["low trunk", "Iz"]])

        self.r_head = Ra.loc["head"][0]

        self.pose = pose
        print("done init")

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

    def update_mass(self, m):
        Ma = m * self.tot_mass

        uarm_m = Ma[3]
        larm_m = Ma[4]
        uleg_m = Ma[5]
        lleg_m = Ma[6]

        # storing arm parameters
        self.arm_m = np.array([uarm_m, larm_m])
        self.leg_m = np.array([uleg_m, lleg_m])
        self.utrunk_m = Ma[1]
        self.ltrunk_m = Ma[2]
        self.m_head = Ma[0]

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

        # for i in [0, 1, 2, 3, 4, 5, 6]:
        # for i in range(3):
        #     Fn = self.Fn[i, j]

        #     Xcop = np.divide(self.Xcop[i, j], Fn)
        #     Ycop = np.divide(self.Ycop[i, j], Fn)
        #     Fn_norm = Fn / self.Fn_tot[j] * 1000
        #     self.ax.quiver(Xcop, Ycop, -Fn_norm, 0, 0, Fn_norm, color="black")
        # print(Fn)
        # print(Xcop, Ycop, -Fn, Xcop, Ycop, 0)
        # Fn_tot = 1000

        # self.ax.quiver(Xcalc[j], Ycalc[j], -Fn_tot, 0, 0, Fn_tot, color="red")

        mid_face = (np.matrix([[x[14]], [y[14]], [z[14]]]) + np.matrix([[x[15]], [y[15]], [z[15]]])) / 2

        # R0f = self.pose.R_xzy(self.pose.thet1f[j], self.pose.thet2f[j], self.pose.thet3f[j])
        # R0f = R.from_matrix(R0f)
        R0f = self.R0f[j]
        R0f = R0f * R.from_euler("x", self.deg_face)
        R0s = self.R0s[j]
        R0h = self.R0h[j]

        T0f = np.zeros((4, 4))
        T0f[0:3, 0:3] = R0f.as_matrix()
        T0f[0:3, 3] = mid_face.reshape(3)
        T0f[3, 3] = 1

        self.ax = self.show_Ti(self.ax, T0f, scale=50)

        center = self.Rhead + self.z_face

        center_proj = mid_face.ravel() + R0f.apply(np.array([0, 0, center * 1000]))
        COP_head = center_proj.copy()
        COP_head[0, 2] = 0

        contact_point = mid_face.ravel() + R0f.apply(np.array([0, 0, (center - self.Rhead) * 1000]))

        cns1 = 1
        cns2 = 2

        plt.plot(center_proj[0, 0], center_proj[0, 1], center_proj[0, 2], "o", color="green")
        plt.plot(contact_point[0, 0], contact_point[0, 1], contact_point[0, 2], "o", color="purple")
        plt.plot(COP_head[0, 0] * cns1, COP_head[0, 1] * cns1, COP_head[0, 2] * cns1, "s", color="black")

        plt.plot(
            [center_proj[0, 0], COP_head[0, 0]],
            [center_proj[0, 1], COP_head[0, 1]],
            [center_proj[0, 2], COP_head[0, 2]],
            "--",
            color="black",
        )

        plt.plot(
            [center_proj[0, 0], contact_point[0, 0]],
            [center_proj[0, 1], contact_point[0, 1]],
            [center_proj[0, 2], contact_point[0, 2]],
            "--",
            color="black",
        )

        mid_hip = (np.matrix([[x[8]], [y[8]], [z[8]]]) + np.matrix([[x[11]], [y[11]], [z[11]]])) / 2
        mid_shoulder = (np.matrix([[x[2]], [y[2]], [z[2]]]) + np.matrix([[x[5]], [y[5]], [z[5]]])) / 2

        Rtrunk = self.ltrunk_w * self.r_trunk_ratio
        z_init = np.mean(self.pose.mid_hip[2, 0:10]) * self.trunk_h_ratio
        center_low = Rtrunk - z_init

        L_trunk = np.mean(np.linalg.norm(self.pose.mid_shoulder - self.pose.mid_hip, axis=0))
        center_up = -L_trunk * self.utrunk_com_ratio

        # Calculating Projection of COM offest
        mid_proj_up = R0s.apply(np.array([0, center_up * 1000, 0]))
        mid_proj_low = R0h.apply(np.array([0, 0, center_low * 1000]))

        # Getting Full COM location by adding offset
        COM_up = (mid_shoulder.T + mid_proj_up).T
        COM_low = (mid_hip.T + mid_proj_low).T

        T0h = np.zeros((4, 4))
        T0h[0:3, 0:3] = R0h.as_matrix()
        T0h[0:3, 3] = mid_hip.reshape(3)
        T0h[3, 3] = 1

        self.ax = self.show_Ti(self.ax, T0h, scale=50)

        T0s = np.zeros((4, 4))
        T0s[0:3, 0:3] = R0s.as_matrix()
        T0s[0:3, 3] = mid_shoulder.reshape(3)
        T0s[3, 3] = 1

        self.ax = self.show_Ti(self.ax, T0s, scale=50)

        contact_point_low = mid_hip.ravel() + R0h.apply(np.array([0, 0, (center_low - Rtrunk) * 1000]))

        plt.plot(COM_low[0, 0], COM_low[1, 0], COM_low[2, 0], "o", color="green")
        plt.plot(contact_point_low[0, 0], contact_point_low[0, 1], contact_point_low[0, 2], "o", color="purple")
        # quit()

        plt.plot(
            [COM_low[0, 0], COM_low[0, 0]],
            [COM_low[1, 0], COM_low[1, 0]],
            [COM_low[0, 0] * 0, COM_low[2, 0]],
            "--",
            color="black",
        )

        plt.plot(
            [COM_low[0, 0], contact_point_low[0, 0]],
            [COM_low[1, 0], contact_point_low[0, 1]],
            [COM_low[2, 0], contact_point_low[0, 2]],
            "--",
            color="black",
        )

        plt.plot(COM_up[0, 0] * cns1, COM_up[1, 0] * cns1, COM_up[2, 0] * 0, "s", color="black")
        plt.plot(COM_low[0, 0] * cns1, COM_low[1, 0] * cns1, COM_low[2, 0] * 0, "s", color="black")

        plt.plot(COM_up[0, 0], COM_up[1, 0], COM_up[2, 0], "o", color="green")
        plt.plot(
            [COM_up[0, 0], COM_up[0, 0]],
            [COM_up[1, 0], COM_up[1, 0]],
            [COM_up[0, 0] * 0, COM_up[2, 0]],
            "--",
            color="black",
        )

        mid = (mid_hip + mid_shoulder) / 2
        mid = mid * 0

        Xcalc = self.Xcalc[0, j] - np.mean(self.Xcalc[0, 0:10] - mid[0, 0])
        Ycalc = self.Ycalc[0, j] - np.mean(self.Ycalc[0, 0:10] - mid[1, 0])

        Xcalc_line = self.Xcalc[0, 0:j] - np.mean(self.Xcalc[0, 0:10] - mid[0, 0])
        Ycalc_line = self.Ycalc[0, 0:j] - np.mean(self.Ycalc[0, 0:10] - mid[1, 0])

        plt.plot(Xcalc * cns2, Ycalc * cns2, 0, "x", color=np.array([255, 0, 30]) / 255)
        plt.plot(Xcalc_line * cns2, Ycalc_line * cns2, 0, color=np.array([255, 0, 30]) / 255, alpha=0.35)

        Xreal = self.COP.Xfilt[::2]
        Yreal = self.COP.Yfilt[::2]

        # print(np.shape(Xreal))
        # quit()

        Xreal_line = Xreal[0:j] - np.mean(Xreal[0:10] - mid[0, 0])
        Yreal_line = Yreal[0:j] - np.mean(Yreal[0:10] - mid[1, 0])

        Xreal = Xreal[j] - np.mean(Xreal[0:10] - mid[0, 0])
        Yreal = Yreal[j] - np.mean(Yreal[0:10] - mid[1, 0])

        plt.plot(Xreal * cns2, Yreal * cns2, 0, "x", color=np.array([10, 35, 175]) / 255)
        plt.plot(Xreal_line * cns2, Yreal_line * cns2, 0, color=np.array([10, 35, 175]) / 255, alpha=0.35)

        s = "t= " + str(int(j / 30))
        self.ax.text(0.3, -0.3, 0, "%s" % (s), size=20, zorder=1, color="k")

        # Head
        self.ax.set_xlim3d(-150, 150)
        self.ax.set_ylim3d(-150, 150)
        self.ax.set_zlim3d(0, 300)
        # # Center
        # self.ax.set_xlim3d(-20, 20)
        # self.ax.set_ylim3d(-120, -80)
        # self.ax.set_zlim3d(0, 40)
        # # Hip
        # self.ax.set_xlim3d(-150, 150)
        # self.ax.set_ylim3d(-300, 0)
        # self.ax.set_zlim3d(0, 300)
        # # Full
        # self.ax.set_xlim3d(-300, 300)
        # self.ax.set_ylim3d(-500, 200)
        # self.ax.set_zlim3d(0, 600)

        plt.grid()

    def plot_cop_anim(self, start=0):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")

        frms = np.linspace(0, self.frames, self.frames + 1, dtype=int)
        # print(frms)

        ani = animation.FuncAnimation(self.fig, self.plot_cop_3d, frames=frms[start:], interval=1)
        plt.show()

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

    def COP_trunk(self):
        pose = self.pose
        g = 9.81

        # lower trunk parameters
        w1, w2 = self.utrunk_w, self.ltrunk_w
        l1, l2 = self.utrunk_l, self.ltrunk_l
        h1, h2 = self.utrunk_h, self.ltrunk_h
        M1, M2 = self.utrunk_m, self.ltrunk_m

        dx_up, dy_up, dz_up = w1 / 2, l1 / 2, h1 / 2
        dx_low, dy_low, dz_low = w2 / 2, l2 / 4, h2 / 2

        Iup = self.Iup
        Ilow = self.Ilow
        Lup = np.mean(pose.get_len(2, 5)) / 2
        Llow = np.mean(pose.get_len(8, 11)) / 2

        mg_up = g * M1
        mg_low = g * M2

        Tqlow, w_low, w_d_low, Vlh_d, Vrh_d = pose.inv_dynamics_lower_vec(Llow, Ilow, to_upper=True)
        Tqup, w_up, w_d_up, Vls_d, Vrs_d = pose.inv_dynamics_upper_vec(Lup, Iup)

        # Right Arm dynamics
        Theta_rarm = self.get_theta(2)
        Trarm, Frarm, e1r, e2r = pose.inv_dynamics_vec(
            Theta_rarm,
            self.arm_len,
            self.arm_m,
            self.arm_I,
            W0=w_up,
            W0_d=w_d_up,
            V0_d=Vrs_d,
            to_upper=self.to_trunk,
        )

        # Left Arm dynamics
        Theta_larm = self.get_theta(5)
        Tlarm, Flarm, e1l, e2l = pose.inv_dynamics_vec(
            Theta_larm,
            self.arm_len,
            self.arm_m,
            self.arm_I,
            W0=w_up,
            W0_d=w_d_up,
            V0_d=Vls_d,
            to_upper=self.to_trunk,
        )

        # Right Leg Dynamics
        Theta_rleg = self.get_theta(8)
        Trleg, Frleg, e1r, e2r = pose.inv_dynamics_vec(
            Theta_rleg,
            self.leg_len,
            self.leg_m,
            self.leg_I,
            W0=w_low,
            W0_d=w_d_low,
            V0_d=Vrh_d,
            to_lower=self.to_trunk,
        )

        # Left Leg Dynamics
        Theta_lleg = self.get_theta(11)
        Tlleg, Flleg, e1l, e2l = pose.inv_dynamics_vec(
            Theta_lleg,
            self.leg_len,
            self.leg_m,
            self.leg_I,
            W0=w_low,
            W0_d=w_d_low,
            V0_d=Vlh_d,
            to_lower=self.to_trunk,
        )

        # print("done ID")

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

        # Getting Fn for upper and lower trunk
        M_up = M1 * np.ones_like(F1z)
        M_low = M2 * np.ones_like(F3z)

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

        # # calculating COP change from upper limbs on upper trunk
        X_calc_upper = np.divide((Ty_up + ((F1z - F2z) * dx_up) + (F12_x * dz_up)), (mg_up - F12_z))
        Y_calc_upper = np.divide((Tx_up - (F12_y * dz_up) + (F12_z * dy_up)), (F12_z - mg_up))

        if self.to_world:
            # Converting Lower Trunk Torque into Upper Trunk ref. frame
            d_up = np.stack((X_calc_upper, Y_calc_upper, np.zeros_like(X_calc_upper)))
            Rs0 = self.R0s.inv()
            d_up_cor = Rs0.apply(d_up.T).T

            X_calc_up = d_up_cor[0, :]
            Y_calc_up = d_up_cor[1, :]
        else:
            X_calc_up = X_calc_upper
            Y_calc_up = Y_calc_upper

        # Paramters for Lower trunk rollovershape
        Rtrunk = w2 * self.r_trunk_ratio
        z_init = np.mean(pose.mid_hip[2, 0:10]) * self.trunk_h_ratio
        center_low = Rtrunk - z_init
        # center_low = self.center_low

        L_trunk = np.mean(np.linalg.norm(pose.mid_shoulder - pose.mid_hip, axis=0))
        # print(l1, L_trunk, L_trunk / l1)
        center_up = -L_trunk * self.utrunk_com_ratio
        # center_up = -self.center_up_y

        self.center_up_y = -center_up
        self.center_up_z = self.center_up_z
        self.center_low = center_low

        # print("Center Up: ", center_up)
        # print("Center Low: ", center_low)

        # deg_trunk = np.deg2rad(30)

        # Calculating Projection of COM offest
        mid_proj_up = self.R0s.apply(np.array([0, center_up, -self.center_up_z]))
        mid_proj_low = self.R0h.apply(np.array([0, 0, center_low]))

        # Getting Full COM location by adding offset
        COM_up = (pose.mid_shoulder.T + mid_proj_up).T
        COM_low = (pose.mid_hip.T + mid_proj_low).T

        # Calculating unweighted upper trunk COP
        X_COM_dCOP = (COM_up[0, :] + (X_calc_up * int(self.include_limbs)))[:, 0 : self.frames]
        Y_COM_dCOP = (COM_up[1, :] + (Y_calc_up * int(self.include_limbs)))[:, 0 : self.frames]

        COP_Xup = np.multiply(X_COM_dCOP, Fn_up)
        COP_Yup = np.multiply(Y_COM_dCOP, Fn_up)

        COP_Xlow = np.multiply(COM_low[0, 0 : self.frames], Fn_low)
        COP_Ylow = np.multiply(COM_low[1, 0 : self.frames], Fn_low)

        if self.plot_dynamics:
            self.plot_ID_LR(Tlarm, Flarm, Trarm, Frarm, title="Arm Dynamics")
            self.plot_ID_LR(Tlleg, Flleg, Trleg, Frleg, title="Leg Dynamics")

        return COP_Xup, COP_Yup, Fn_up, COP_Xlow, COP_Ylow, Fn_low

    def COP_head(self):
        pose = self.pose

        head_h_ratio_chest = self.head_h_ratio_chest
        r_head_ratio = self.r_head_ratio
        deg_face = self.deg_face

        Fn_head = (self.m_head) * np.ones((1, self.frames)) * 9.81

        w_head = np.mean(pose.get_len(16, 17)) * r_head_ratio
        Rhead = w_head

        nose = np.array([pose.X[:, 0], pose.Y[:, 0], pose.Z[:, 0]])

        # d_ears_eyes = np.mean(np.linalg.norm(pose.mid_face - pose.mid_ear, axis=0))
        # z_head = d_ears_eyes

        z_head_from_chest = -(np.mean(pose.Z[1, 0:300])) * head_h_ratio_chest

        # Rhead = 0.098447
        # z_head_from_chest = -0.13584
        center = Rhead + z_head_from_chest
        # center = self.center_head
        self.center_head = center

        # print("Center Head: ", center)

        # print(w_head, center, self.r_head, Rhead, (self.r_head - z_head_from_chest))

        Rface = self.R0f
        Rface = Rface * R.from_euler("x", deg_face)

        if self.include_head:
            mid_proj_face = Rface.apply(np.array([0, 0, center]))

            a = Rhead
            b = Rhead
            c = Rhead

            # contact = self.ellipsoid(Rface, a, b, c)
            # zoff = z_head_from_chest

            # ellipsoid_center = Rface.apply(np.array([0, 0, zoff]))
            # COM_head = (pose.mid_face.T + ellipsoid_center + contact).T
        else:
            mid_proj_face = Rface.apply(np.array([0, 0, 0]))

        COM_head = (pose.mid_face.T + mid_proj_face).T
        # print(np.shape(COM_head), np.shape(Fn_head), self.frames)

        self.Rhead = Rhead
        self.z_face = z_head_from_chest

        # a = 4 / 1000
        # b = 4 / 1000
        # c = Rhead / 5

        # contact = self.ellipsoid(Rface, a, b, c)

        # ellipsoid_center = Rface.apply(np.array([0, d_nose_eyes, (Rhead - z_chest)]))
        # COM_head = (pose.mid_face.T + ellipsoid_center + contact).T

        # print(pose.mid_face[:, 0] * 1000, ellipsoid_center[0, :].T * 1000, contact[0, :].T * 1000)
        # print(np.shape(COM_head), np.shape(Fn_head), self.frames)

        COP_head = np.multiply(COM_head[:, 0 : self.frames], Fn_head[0 : self.frames])
        X_head = COP_head[0, :]
        Y_head = COP_head[1, :]

        # plt.plot(np.rad2deg(pose.thet1f.T))
        # plt.plot(np.rad2deg(pose.thet2f.T))
        # plt.plot(np.rad2deg(pose.thet3f.T))
        # plt.show()

        # # print(np.shape(self.pose.mid_ear))

        # side = np.where(abs(np.rad2deg(pose.thet3f)) >= (theta_range / 2))[1]
        # X_head[0, side] = self.pose.mid_ear[0, side] * self.m_head * 9.81
        # Y_head[0, side] = self.pose.mid_ear[1, side] * self.m_head * 9.81

        return X_head, Y_head, Fn_head

    def ellipsoid(self, Rot, a, b, c):

        # Rotating vector normal to the ground out of world frame
        norm_rot = Rot.inv().apply(np.matrix([0, 0, -1]))
        # Finding the vector from the roated contact vector to the ellipse
        norm_ellipse = np.multiply(norm_rot, -np.array([a**2 / 2, b**2 / 2, c**2 / 2]))
        # Rotating to world coordinates
        contact = Rot.apply(norm_ellipse)

        # Solving for theta and phi of vector from ellipse center to contact point
        x, y, z = contact[:, 0], contact[:, 1], contact[:, 2]
        theta = np.atan2(y, x)
        phi = np.atan2((np.sqrt(x**2 + y**2)), z)

        # Finding magnitude of radius at contact point
        p1 = (b**2) * (c**2) * (np.cos(theta)) ** 2 * (np.cos(phi)) ** 2
        p2 = (a**2) * (c**2) * (np.sin(theta)) ** 2 * (np.cos(phi)) ** 2
        p3 = (a**2) * (b**2) * (np.sin(phi)) ** 2
        mag = (a * b * c) / np.sqrt(p1 + p2 + p3)

        # print(mag * 1000)

        return np.multiply(mag, contact.T).T

    def calc_COP(self):
        X_up, Y_up, Fn_up, X_low, Y_low, Fn_low = self.COP_trunk()

        X_head, Y_head, Fn_head = self.COP_head()

        ##UPPER/LOWER TRUNK
        Fn_tot = Fn_up + Fn_low + Fn_head

        self.Xcalc = (X_low + X_up + X_head) / Fn_tot * 1000
        self.Ycalc = (Y_low + Y_up + Y_head) / Fn_tot * 1000

        # self.Xcalc = X_low / Fn_tot * 1000
        # self.Fn = np.array([Fn_low.ravel(), Fn_up.ravel(), Fn_head.ravel()])
        # self.Xcop = np.array([X_low.ravel(), X_up.ravel(), X_head.ravel()]) * 1000
        # self.Ycop = np.array([Y_low.ravel(), Y_up.ravel(), Y_head.ravel()]) * 1000

        # t = np.linspace(0, (self.frames) / 30, num=self.frames)
        # plt.plot(t, X_head.T)
        # plt.plot(t, X_up.T)
        # plt.plot(t, X_low.T)
        # # plt.plot(t, (X_low + X_up + X_head).T)
        # plt.grid()
        # plt.legend(["Head", "Up", "Low", "Total"])
        # plt.show()

        # # self.corr_plot()

        # # self.cor_COP()
        # # self.path_len()
        # quit()

        self.Fn_tot = Fn_tot

    def corr_plot(self):
        Xreal = self.COP.Xfilt[::2]
        Yreal = self.COP.Yfilt[::2]

        n = min(self.frames, Xreal.shape[0])

        Xcop = self.Xcop[:, 0, 0:n]
        Ycop = self.Ycop[:, 0, 0:n]

        Xlow, Xup, Xhead = (
            Xcop[0] / (np.max(Xcop[0]) - np.min(Xcop[0])),
            Xcop[1] / (np.max(Xcop[1]) - np.min(Xcop[1])),
            Xcop[2] / (np.max(Xcop[2]) - np.min(Xcop[2])),
        )
        Ylow, Yup, Yhead = (
            Ycop[0] / (np.max(Ycop[0]) - np.min(Ycop[0])),
            Ycop[1] / (np.max(Ycop[1]) - np.min(Ycop[1])),
            Ycop[2] / (np.max(Ycop[2]) - np.min(Ycop[2])),
        )

        Xtot = Xreal[0:n] / (np.max(Xreal) - np.min(Xreal))
        Ytot = Yreal[0:n] / (np.max(Yreal) - np.min(Yreal))

        t = np.linspace(0, n / 30, num=n)

        # print(np.shape(Xhead), np.shape(t), np.shape(Xtot))
        # quit()

        chead = "green"
        cup = "red"
        clow = "blue"
        ctot = "orange"

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 5), sharex=True)

        # plt.subplot(3, 2, 1)
        ax1.plot(t, (Xhead.ravel() - np.mean(Xhead[0:10])) / 1000, color=chead)
        ax1.plot(t, (Xtot.ravel() - np.mean(Xtot[0:10])) / 1000, color=ctot)
        ax1.legend(["Head", "Total"])
        ax1.set_title("X Head")
        ax1.set(ylabel="COP X (mm)", xlabel="Time (s)")
        ax1.grid()

        ax3.plot(t, (Xup.ravel() - np.mean(Xup[0:10])) / 1000, color=cup)
        ax3.plot(t, (Xtot.ravel() - np.mean(Xtot[0:10])) / 1000, color=ctot)
        ax3.legend(["Up", "Total"])
        ax3.set_title("X Up")
        ax3.set(ylabel="COP X (mm)", xlabel="Time (s)")
        ax3.grid()

        ax5.plot(t, (Xlow.ravel() - np.mean(Xlow[0:10])) / 1000, color=clow)
        ax5.plot(t, (Xtot.ravel() - np.mean(Xtot[0:10])) / 1000, color=ctot)
        ax5.legend(["Low", "Total"])
        ax5.set_title("X Low")
        ax5.set(ylabel="COP X (mm)", xlabel="Time (s)")
        ax5.grid()

        ax2.plot(t, (Yhead.ravel() - np.mean(Yhead[0:10])) / 1000, color=chead)
        ax2.plot(t, (Ytot.ravel() - np.mean(Ytot[0:10])) / 1000, color=ctot)
        ax2.legend(["Head", "Total"])
        ax2.set_title("Y Head")
        ax2.set(ylabel="COP Y (mm)", xlabel="Time (s)")
        ax2.grid()

        ax4.plot(t, (Yup.ravel() - np.mean(Yup[0:10])) / 1000, color=cup)
        ax4.plot(t, (Ytot.ravel() - np.mean(Ytot[0:10])) / 1000, color=ctot)
        ax4.legend(["Up", "Total"])
        ax4.set_title("Y Up")
        ax4.set(ylabel="COP Y (mm)", xlabel="Time (s)")
        ax4.grid()

        ax6.plot(t, (Ylow.ravel() - np.mean(Ylow[0:10])) / 1000, color=clow)
        ax6.plot(t, (Ytot.ravel() - np.mean(Ytot[0:10])) / 1000, color=ctot)
        ax6.legend(["Low", "Total"])
        ax6.set_title("Y Low")
        ax6.set(ylabel="COP Y (mm)", xlabel="Time (s)")
        ax6.grid()

        plt.tight_layout()
        plt.show()

    def corr_COP(self):
        Xreal = self.COP.Xfilt[::2]
        Yreal = self.COP.Yfilt[::2]

        n = min(self.frames, Xreal.shape[0])

        xlow, xup, xhead = self.Xcop[0, 0, 0:n], self.Xcop[1, 0, 0:n], self.Xcop[2, 0, 0:n]
        ylow, yup, yhead = self.Ycop[0, 0, 0:n], self.Ycop[1, 0, 0:n], self.Ycop[2, 0, 0:n]

        Xreal = Xreal[0:n]
        Yreal = Yreal[0:n]

        corr_x_head, _ = stats.pearsonr(xhead, Xreal)
        corr_x_up, _ = stats.pearsonr(xup, Xreal)
        corr_x_low, _ = stats.pearsonr(xlow, Xreal)

        corr_y_head, _ = stats.pearsonr(yhead, Yreal)
        corr_y_up, _ = stats.pearsonr(yup, Yreal)
        corr_y_low, _ = stats.pearsonr(ylow, Yreal)

        # print("Corr X: Head ", corr_x_head, " Up ", corr_x_up, " Low ", corr_x_low)
        # print("Corr Y: Head ", corr_y_head, " Up ", corr_y_up, " Low ", corr_y_low)

        data = {
            "Corr X Head": corr_x_head,
            "Corr X Up": corr_x_up,
            "Corr X Low": corr_x_low,
            "Corr Y Head": corr_y_head,
            "Corr Y Up": corr_y_up,
            "Corr Y Low": corr_y_low,
        }
        df = pd.DataFrame(data, index=[0])

        # print(data)

        return df

    def path_len(self):
        pose = self.pose

        head = pose.mid_face
        up = pose.mid_shoulder
        low = pose.mid_hip

        path_head = np.linalg.norm(head[:, 0:-2] - head[:, 1:-1], axis=0)
        path_up = np.linalg.norm(up[:, 0:-2] - up[:, 1:-1], axis=0)
        path_low = np.linalg.norm(low[:, 0:-2] - low[:, 1:-1], axis=0)

        data = {
            "Path Head": np.sum(path_head),
            "Path Up": np.sum(path_up),
            "Path Low": np.sum(path_low),
        }
        df = pd.DataFrame(data, index=[0])

        return df

    def rad_metrics(self):
        pose = self.pose

        # Head
        w_head = np.mean(pose.get_len(16, 17))
        Rhead = w_head / (2 * np.sin(np.deg2rad(self.thet_range_head)))
        z_chest = np.mean(pose.Z[1, :]) * 2

        head_center = pose.mid_face.T + self.R0f.apply(np.array([0, 0, (Rhead - z_chest)]))
        head_cop = head_center.copy()
        head_cop[:, 2] = 0

        head_dist = np.mean(np.linalg.norm((head_cop - head_center), axis=1))

        # Lower Trunk
        Rtrunk = self.ltrunk_w / (2 * np.sin(np.deg2rad(self.thet_range_low)))
        z_init = np.mean(pose.mid_hip[2, 0:10])

        # Calculating Projection of COM offest
        low_center = pose.mid_hip.T + self.R0h.apply(np.array([0, 0, (Rtrunk - z_init)]))
        low_cop = low_center.copy()
        low_cop[:, 2] = 0

        low_dist = np.mean(np.linalg.norm((low_cop - low_center), axis=1))

        data = {
            "R Head": Rhead * 1000,
            "Z Head": z_chest * 1000,
            "Diff Head": (Rhead - z_chest) * 1000,
            "Dist Head": head_dist * 1000,
            "R Low": Rtrunk * 1000,
            "Z Low": z_init * 1000,
            "Diff Low": (Rtrunk - z_init) * 1000,
            "Dist Low": low_dist * 1000,
        }
        df = pd.DataFrame(data, index=[0])

        # print(df)

        return df

    def load_compare(self, start=0, stop=-1, Xreal=[], Yreal=[], offset=0):
        Xcalc = self.Xcalc.T
        Ycalc = self.Ycalc.T

        if len(Xreal) == 0:
            Xreal = self.COP.Xfilt[::2]
        if len(Yreal) == 0:
            Yreal = self.COP.Yfilt[::2]

        # Offset
        if offset == 0:
            Xreal, Yreal = Xreal[start:stop], Yreal[start:stop]
            Xcalc, Ycalc = Xcalc[start:stop], Ycalc[start:stop]
        elif offset > 0:
            Xreal, Yreal = Xreal[start + offset : stop], Yreal[start + offset : stop]
        elif offset < 0:
            # print(start, start - offset)
            Xcalc, Ycalc = Xcalc[start - offset : stop], Ycalc[start - offset : stop]

        win = 2

        Xcalc = scipy.ndimage.median_filter(Xcalc, win)
        Ycalc = scipy.ndimage.median_filter(Ycalc, win)

        compare_object = compareCOP(Xcalc, Ycalc, Xreal, Yreal, cam=1)

        return compare_object

    def compare_COP(self, start=60, stop=-1, offset=0):
        compare_object = self.load_compare(start=start, stop=stop, offset=offset)

        compare_object.comp_XY()
        compare_object.comp_ellipse()
        # cc.plot_cop_anim()

        # print(cc.metrics())
        print(compare_object.diff_metric())

    def update_params(self, x):

        # self.head_h_ratio_chest = x[0]
        # self.r_head_ratio = x[1]
        # self.r_trunk_ratio = x[2]
        # self.trunk_h_ratio = x[3]

        # self.a_head = x[0]
        # self.b_head = x[1]
        # self.c_head = x[2]
        # self.z_off_head = x[3]

        # M = x[4:11]
        # self.update_mass(m=M)
        self.center_head = x[0]
        self.center_up_y = x[1]
        self.center_up_z = x[2]
        self.center_low = x[3]

        # print("param updated")

    def residual(self, x):
        self.update_params(x)
        self.calc_COP()

        Xcalc = self.Xcalc.T
        Ycalc = self.Ycalc.T

        Xreal = self.COP.Xfilt[::2]
        Yreal = self.COP.Yfilt[::2]

        win = 2

        Xcalc = scipy.ndimage.median_filter(Xcalc, win)
        Ycalc = scipy.ndimage.median_filter(Ycalc, win)

        n = min(Xcalc.shape[0], Xreal.shape[0])

        Xcalc = Xcalc[0:n].reshape((n, 1))
        Ycalc = Ycalc[0:n].reshape((n, 1))

        Xreal = Xreal[0:n].reshape((n, 1))
        Yreal = Yreal[0:n].reshape((n, 1))

        Xr = Xreal - np.mean(Xreal[0:15])
        Yr = Yreal - np.mean(Yreal[0:15])

        Xc = Xcalc - np.mean(Xcalc[0:15])
        Yc = Ycalc - np.mean(Ycalc[0:15])

        val_X, p_X = stats.pearsonr(Xr, Xc, axis=0)
        val_Y, p_Y = stats.pearsonr(Yr, Yc, axis=0)

        rX, rY = Xr - Xc, Yr - Yc

        # m = x[4:11]
        # mass_sum = sum(m[3:7]) + sum(m)
        # mass_sum_loss = 1 - mass_sum

        # self.m_dist = m

        # res = np.hstack((rX.ravel() * abs(1 - val_X), rY.ravel() * abs(1 - val_Y), [mass_sum_loss]))
        res = np.hstack((rX.ravel() * abs(1 - val_X), rY.ravel() * abs(1 - val_Y)))
        # res = (rX.ravel() * abs(1 - val_X)) + (rY.ravel() * abs(1 - val_Y))
        return res

    def optim_COP(self, start=60, stop=-1):
        self.calc_COP()
        # self.compare_object = self.load_compare(start=start, stop=stop)

        # r0 = np.array(
        #     [
        #         self.head_h_ratio_chest,
        #         self.r_head_ratio,
        #         self.r_trunk_ratio,
        #         self.trunk_h_ratio,
        #     ]
        # )

        # low_bounds = np.array([0, 0, 0, 0])
        # up_bounds = np.array([5, 5, 5, 5])

        r0 = np.array([self.center_head, self.center_up_y, self.center_up_z, self.center_low])

        low_bounds = np.array([-0.5, 0, 0, -0.5])
        up_bounds = np.array([0.5, 0.75, 0.5, 0.5])

        # r0 = np.array(
        #     [
        #         self.a_head,
        #         self.b_head,
        #         self.c_head,
        #         self.z_off_head,
        #     ]
        # )
        m0 = self.m_dist
        # print(len(m0))
        # print("Total Mass: ", sum(m0[3:7]) + sum(m0))
        # quit()

        m0 = []

        # self.head_h_ratio_chest = 1.5
        # self.r_head_ratio = 0.5
        # self.r_trunk_ratio = 0.5
        # self.trunk_h_ratio = 0.75
        x0 = np.concat((r0, m0))
        print("Initial Radius Terms", x0[0:4])
        print("Initial Mass distribution", x0[4:11])

        # self.utrunk_com_ratio = 0.25

        low_bounds_mass = np.array([0.1, 0.1, 0.1, 0.005, 0.005, 0.005, 0.005])
        up_bounds_mass = np.array([0.75, 0.75, 0.75, 0.5, 0.6, 0.5, 0.6])

        low_bounds_mass = []
        up_bounds_mass = []

        res = least_squares(
            self.residual,
            x0,
            verbose=2,
            ftol=1e-3,
            bounds=(np.concat((low_bounds, low_bounds_mass)), np.concat((up_bounds, up_bounds_mass))),
        )

        finalx = res.x
        finalm = finalx[4:11]
        print("Final Radius Terms", finalx[0:4])
        print("Final Mass distribution", finalm)
        print("Total Mass: ", sum(finalm[4:7]) + sum(finalm))
        # print(res.x - x0)

        self.compare_COP()
