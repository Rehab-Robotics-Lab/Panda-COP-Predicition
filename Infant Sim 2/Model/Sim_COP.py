from Inv_dynamics import inv_dynamics
from ProcessCOP import processCOP
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import constants


class sim_COP:
    # mass=[2X1], length=[2x1], I=[2x3], theta=[4,t]
    def __init__(self):
        robofile = pd.read_csv(r"C:\Users\franc\Documents\Infant_Sim_data\load tests\lleg_101.csv")
        # LOADING FILE WITH COP VALUES
        cop_file = r"C:\Users\franc\Documents\Infant_Sim_data\load tests\cop_lleg_101.csv"
        self.rate = 60
        # cop object
        self.COP = processCOP(cop_file, self.rate)

        ##INITIALIZING PARAMETERS
        # Upper Trunk parameters
        self.utrunk_l = 0.13
        self.utrunk_w = 0.2
        self.utrunk_h = 0.1
        self.utrunk_m = 2.312 - (0.373 + 0.345)

        # Lower Trunk parameters
        self.ltrunk_l = 0.11
        self.ltrunk_w = 0.18
        self.ltrunk_h = 0.1
        self.ltrunk_m = 2.9 - (0.701 + 0.677)

        # Arm paramters
        # lengths
        uarm_len = 0.1
        larm_len = 0.080
        # radius
        uarm_r = 0.20
        larm_r = 0.15
        # mass
        l_uarm_m = 0.2
        l_larm_m = 0.373 - l_uarm_m
        r_uarm_m = 0.2
        r_larm_m = 0.345 - r_uarm_m
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
        uleg_len = 0.110
        lleg_len = 0.10
        # radius
        uleg_r = 0.6
        lleg_r = 0.30
        # mass
        l_uleg_m = 0.5
        l_lleg_m = 0.756 - l_uleg_m
        r_uleg_m = 0.5
        r_lleg_m = 0.756 - r_uleg_m
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

        # initializing values from robot kinematics file

        self.robofile = robofile

        # left arm angles from robot file
        arm_angle_l = robofile.larm_angle
        arm_angle_r = robofile.rarm_angle
        leg_angle_l = robofile.lleg_angle
        leg_angle_r = robofile.rleg_angle

        # putting angles from robot in fromat with angles from all limbs
        self.left_arm_angles = self.sim_to_theta_arm(arm_angle_l.to_numpy())
        self.right_arm_angles = self.sim_to_theta_arm(arm_angle_r.to_numpy())
        self.left_leg_angles = self.sim_to_theta_leg(leg_angle_l.to_numpy() + 110)
        self.right_leg_angles = self.sim_to_theta_leg(leg_angle_r.to_numpy() + 110)

        self.robot_t = robofile.time.to_numpy()
        # time step between input angles
        self.dt = np.mean(np.diff(self.robot_t[0:30]))

        # checking when left arm is in contact with the ground
        self.larm_check = arm_angle_l == -45
        self.rarm_check = arm_angle_r == -45
        self.lleg_check = leg_angle_l == -100
        self.rleg_check = leg_angle_r == -100

        self.larm_Ld = robofile.larm_load * 1.4 / 1000
        self.rarm_Ld = robofile.rarm_load * 1.9 / 1000
        self.lleg_Ld = robofile.lleg_load * 1.9 / 1000
        self.rleg_Ld = robofile.rleg_load * 1.9 / 1000

    # function converts angles from infant simulator to angles for RNE/FK
    def sim_to_theta_arm(self, angle):
        # array of all angles
        theta_arm = np.zeros((4, len(angle)))

        # theta 2 is angles from robot and theta 4 is fixed elbow angle of about 30 deg
        # theta_arm[0, :] = np.ones_like(angle) * 10
        theta_arm[1, :] = angle
        theta_arm[3, :] = np.ones_like(angle) * -90

        return theta_arm

    def sim_to_theta_leg(self, angle):
        # array of all angles
        theta_leg = np.zeros((4, len(angle)))

        # theta 2 is angles from robot and theta 4 is fixed elbow angle of about 30 deg
        # theta_leg[0, :] = np.ones_like(angle) * 45
        theta_leg[1, :] = angle
        theta_leg[2, :] = np.ones_like(angle) * -45
        theta_leg[3, :] = np.ones_like(angle) * 90

        return theta_leg

    def I(self, l, m, r):
        I1 = 1 / 2 * (m * r**2)
        I2 = 1 / 12 * (m * (3 * r**2 + l**2))
        # interia tensor of cylinder(positions of I1 and I2 depend on definition of axes)
        # b/c of DH param x axis should always be along cylinder length
        return np.array([I1, I2, I2])

    def COP_upper(self):
        # right arm dynamics
        rarm_dynamics = inv_dynamics(self.right_arm_angles, self.rarm_len, self.rarm_m, self.rarm_I, self.dt)
        Trarm, Frarm = rarm_dynamics.calc()
        Trarm[:, self.rarm_check] = 0
        Frarm[:, self.rarm_check] = 0
        # Wrinting dynamic terms for right arm
        T1x = Trarm[0, :]
        T1y = Trarm[1, :]

        F1x = Frarm[0, :]
        F1y = Frarm[1, :]
        F1z = Frarm[2, :]

        # left arm dynamics
        yflip = np.matrix([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

        larm_dynamics = inv_dynamics(self.left_arm_angles, self.larm_len, self.larm_m, self.larm_I, self.dt)
        Tlarm, Flarm = larm_dynamics.calc()
        Tlarm[:, self.larm_check] = 0
        Flarm[:, self.larm_check] = 0

        Tlarm = yflip @ Tlarm
        Flarm = yflip @ Flarm
        # Wrinting dynamic terms for left arm
        # T2x = -Tlarm[0, :]
        # T2y = -Tlarm[1, :]

        # F2x = Flarm[0, :]
        # F2y = -Flarm[1, :]
        # F2z = -Flarm[2, :]

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

        g = 9.81

        # X_calc = (F2 * 0.5 * w) / (F2 + g * M)
        # Y_calc = ((F2 * l / 2) - T2) / (F2 + g * M)
        a, b, c, Tt = self.COP_lower()

        # X_calc = ((T1y - T2y - Tt) + ((F1x - F2x) * h * 0.5) + ((F2z - F1z) * w * 0.5)) / (F1z + F2z + (g * M))
        # Y_calc = (((F1z + F2z) * l * 0.5) - ((F1y + F2y) * h * 0.5) - (T1x + T2x)) / (F1z + F2z + (g * M))

        X_calc = ((T1y + T2y + Tt) + ((F1x - F2x) * h * 0.5) + ((F1z - F2z) * w * 0.5)) / (-F1z - F2z + (g * M))
        Y_calc = ((T1x - T2x) + ((F1z + F2z) * l * 0.5) - ((F1y + F2y) * h * 0.5)) / (F1z + F2z - (g * M))

        X_calc = X_calc * 1000
        Y_calc = Y_calc * 1000

        Fn = (g * M * np.ones_like(X_calc)) - F1z - F2z

        return X_calc, Y_calc, Fn

    def COP_lower(self):

        thet = np.deg2rad(30)
        rotz = np.matrix([[np.cos(thet), -np.sin(thet), 0], [np.sin(thet), np.cos(thet), 0], [0, 0, 1]])

        yflip = np.matrix([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # left leg dynamics
        lleg_dynamics = inv_dynamics(self.left_leg_angles, self.lleg_len, self.lleg_m, self.lleg_I, self.dt)
        Tlleg, Flleg = lleg_dynamics.calc()

        # rotating torque and force values by 30 degrees
        Tlleg = rotz @ Tlleg
        Flleg = rotz @ Flleg

        Tlleg = yflip @ Tlleg
        Flleg = yflip @ Flleg

        # zeroing force and torque when leg is at rest
        Tlleg[:, self.lleg_check] = 0
        Flleg[:, self.rarm_check] = 0

        # right leg dynamics
        rleg_dynamics = inv_dynamics(self.right_leg_angles, self.rleg_len, self.rleg_m, self.rleg_I, self.dt)
        Trleg, Frleg = rleg_dynamics.calc()
        # rotating torque and force values by 30 degrees
        Trleg = rotz @ Trleg
        Frleg = rotz @ Frleg
        # zeroing force and torque when leg is at rest
        Trleg[:, self.rleg_check] = 0
        Frleg[:, self.rarm_check] = 0

        # right leg dynamic terms
        T3x = Trleg[0, :]
        T3y = Trleg[1, :]

        F3x = Frleg[0, :]
        F3y = Frleg[1, :]
        F3z = Frleg[2, :]

        # left leg dynamic terms
        T4x = Tlleg[0, :]
        T4y = Tlleg[1, :]

        F4x = Flleg[0, :]
        F4y = Flleg[1, :]
        F4z = Flleg[2, :]

        # lower trunk parameters
        l = self.ltrunk_l
        w = self.ltrunk_w
        M = self.ltrunk_m
        h = self.ltrunk_h

        g = 9.81

        # plt.plot(T4x.T)
        # plt.show()
        # exit()

        # Y_calc = (((F3z + F4z) * l * 0.5) - ((F3y + F4y) * h * 0.5) - (T3x + T4x)) / (F3z + F3z + (g * M))
        X_calc = np.zeros_like(F3y)
        # Y_calc = ((T3x + T4x) + ((F3y + F4y) * h * 0.5) + ((F3z + F4z) * l * 0.5)) / (F3z + F4z + (g * M))
        Y_calc = ((T3x - T4x) - ((F3y + F4y) * h * 0.5) - ((F3z + F4z) * l * 0.5)) / (F3z + F4z - (g * M))

        X_calc = X_calc * 1000
        Y_calc = Y_calc * 1000

        T = (T4y + T3y) + ((F3z - F4z) * w * 0.5) + ((F3x - F4x) * h * 0.5)

        Fn = (g * M * np.ones_like(F3z)) - F3z - F4z

        return X_calc, Y_calc, Fn, T

    def calc_COP(self):

        X_up, Y_up, Fn_up = self.COP_upper()
        X_low, Y_low, Fn_low, Tt = self.COP_lower()

        # Lower trunk offset in y
        ytrunk = (self.utrunk_l / 2) + (self.ltrunk_l / 2)
        Y_low = -ytrunk + Y_low

        # reshaping to make sure all dimenesions are [1,n]
        n = len(self.lleg_check)

        larm_check = np.asarray(self.larm_check).reshape(1, n)
        rarm_check = np.asarray(self.rarm_check).reshape(1, n)
        lleg_check = np.asarray(self.lleg_check).reshape(1, n)
        rleg_check = np.asarray(self.rleg_check).reshape(1, n)

        # resting COP for arms
        xarm = (self.utrunk_w / 2) + 0.06
        yarm = (self.utrunk_l / 2) - 0.1

        X_larm = xarm * larm_check
        Y_larm = yarm * larm_check
        Fn_larm = np.sum(self.larm_m) * 9.81 * larm_check

        X_rarm = -xarm * rarm_check
        Y_rarm = yarm * rarm_check
        Fn_rarm = np.sum(self.rarm_m) * 9.81 * rarm_check

        # resting COP for legs
        xleg = (self.ltrunk_w / 2) + 0.02
        yleg = (self.ltrunk_l / 2) - 0.13 - ytrunk

        X_lleg = xleg * lleg_check
        Y_lleg = yleg * lleg_check
        Fn_lleg = np.sum(self.lleg_m) * 9.81 * lleg_check

        X_rleg = -xleg * rleg_check
        Y_rleg = yleg * rleg_check
        Fn_rleg = np.sum(self.rleg_m) * 9.81 * rleg_check

        # self.Xcalc = X_low
        # self.Ycalc = Y_low
        self.Xcalc = (
            np.multiply(X_low, Fn_low)
            + np.multiply(X_up, Fn_up)
            + np.multiply(X_larm, Fn_larm)
            + np.multiply(X_rarm, Fn_rarm)
            + np.multiply(X_lleg, Fn_lleg)
            + np.multiply(X_rleg, Fn_rleg)
        ) / (Fn_larm + Fn_up + Fn_rarm + Fn_low + Fn_lleg + Fn_rleg)

        self.Ycalc = (
            np.multiply(Y_low, Fn_low)
            + np.multiply(Y_up, Fn_up)
            + np.multiply(Y_larm, Fn_larm)
            + np.multiply(Y_rarm, Fn_rarm)
            + np.multiply(Y_lleg, Fn_lleg)
            + np.multiply(Y_rleg, Fn_rleg)
        ) / (Fn_larm + Fn_up + Fn_rarm + Fn_low + Fn_lleg + Fn_rleg)

    def compare_T(self):
        t = self.robot_t

        larm_dynamics = inv_dynamics(self.left_arm_angles, self.larm_len, self.larm_m, self.larm_I, self.dt)
        Tlarm, Flarm = larm_dynamics.calc()
        Tlarm[:, self.larm_check] = 0
        plt.subplot(2, 2, 1)
        plt.plot(t, self.larm_Ld)
        plt.plot(t, Tlarm[0, :])
        plt.legend(["Grnd Trth", "Calculated"])
        plt.title("L arm Torque")

        rarm_dynamics = inv_dynamics(self.right_arm_angles, self.rarm_len, self.rarm_m, self.rarm_I, self.dt)
        Trarm, Frarm = rarm_dynamics.calc()
        Trarm[:, self.rarm_check] = 0
        plt.subplot(2, 2, 2)
        plt.plot(t, self.rarm_Ld)
        plt.plot(t, Trarm[0, :])
        plt.legend(["Grnd Trth", "Calculated"])
        plt.title("R arm Torque")

        lleg_dynamics = inv_dynamics(self.left_leg_angles, self.lleg_len, self.lleg_m, self.lleg_I, self.dt)
        Tlleg, Flleg = lleg_dynamics.calc()
        Tlleg[:, self.lleg_check] = 0
        plt.subplot(2, 2, 3)
        # plt.plot(t, self.lleg_Ld)
        plt.plot(t, Tlleg[0, :])
        plt.plot(t, self.lleg_Ld)
        plt.legend(["Grnd Trth", "Calculated"])
        plt.title("L leg Torque")

        rleg_dynamics = inv_dynamics(self.right_leg_angles, self.rleg_len, self.rleg_m, self.rleg_I, self.dt)
        Trleg, Frleg = rleg_dynamics.calc()
        Trleg[:, self.rleg_check] = 0
        plt.subplot(2, 2, 4)
        plt.plot(t, self.rleg_Ld)
        plt.plot(t, Trleg[0, :])
        plt.legend(["Grnd Trth", "Calculated"])
        plt.title("R leg Torque")

        plt.show()

    def compare_COP(self):
        n = len(self.robot_t)

        t = np.asarray(self.robot_t).reshape(n, 1)

        X, Y = self.COP.Xfilt, self.COP.Yfilt
        t_c = np.linspace(0, (len(X) - 1) / self.rate, num=len(X))

        # print(np.max(self.Xcalc) - np.min(self.Xcalc))
        # print(np.max(self.Ycalc) - np.min(self.Ycalc))

        Xcalc = self.Xcalc.reshape(n, 1)
        Ycalc = self.Ycalc.reshape(n, 1)

        plt.subplot(2, 1, 1)
        plt.plot(t_c + t[0], X - np.mean(X))
        plt.plot(t, Xcalc - np.mean(Xcalc))
        plt.legend(["Grnd Trth", "Calculated"])
        plt.title("X COP")

        plt.subplot(2, 1, 2)
        plt.plot(t_c + t[0], Y - np.mean(Y))
        plt.plot(t, Ycalc - np.mean(Ycalc))
        plt.legend(["Grnd Trth", "Calculated"])
        plt.title("Y COP")
        plt.show()


test = sim_COP()
test.calc_COP()
test.compare_COP()
# test.compare_T()
