from Inv_dynamics import inv_dynamics
from ProcessCOP import processCOP
from ProcessPose import processPose
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import constants


class sim_COP:
    # mass=[2X1], length=[2x1], I=[2x3], theta=[4,t]
    def __init__(self):

        robofile = r"C:\Users\franc\Documents\Infant_Sim_data\load tests\side_bend_35.pkl"
        self.Pose = processPose(robofile)

        self.Pose.trunk_com()

        # LOADING FILE WITH COP VALUES
        cop_file = r"C:\Users\franc\Documents\Infant_Sim_data\COP\2024-2-28_4_36_Side_bend_35_10_4.csv"
        self.rate = 60
        # cop object
        self.COP = processCOP(cop_file, self.rate)

        ##INITIALIZING PARAMETERS
        # Upper Trunk parameters
        self.utrunk_l = 0.13
        self.utrunk_w = 0.2
        self.utrunk_h = 0.1
        self.utrunk_m = 1.251 + 0.373 + 0.345

        # Lower Trunk parameters
        self.ltrunk_l = 0.11
        self.ltrunk_w = 0.18
        self.ltrunk_h = 0.1
        self.ltrunk_m = 1.532 + 0.677 + 0.701

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
        r_uarm_m = 0.22
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
        uleg_r = 0.65
        lleg_r = 0.4
        # mass
        l_uleg_m = 0.4
        l_lleg_m = 0.748 - l_uleg_m
        r_uleg_m = 0.4
        r_lleg_m = 0.783 - r_uleg_m
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

        self.M_head = 0.945

        # initializing values from robot kinematics file

    # function converts angles from infant simulator to angles for RNE/FK
    def sim_to_theta_arm(self, angle):
        # array of all angles
        theta_arm = np.zeros((4, len(angle)))

        # theta 2 is angles from robot and theta 4 is fixed elbow angle of about 30 deg
        # theta_arm[0, :] = np.ones_like(angle) * 10
        theta_arm[1, :] = angle
        theta_arm[2, :] = np.ones_like(angle) * -30
        theta_arm[3, :] = np.ones_like(angle) * 90

        return theta_arm

    def sim_to_theta_leg(self, angle):
        # array of all angles
        theta_leg = np.zeros((4, len(angle)))

        # theta 2 is angles from robot and theta 4 is fixed elbow angle of about 30 deg
        # theta_leg[0, :] = np.ones_like(angle) * -20
        theta_leg[1, :] = angle
        theta_leg[2, :] = np.ones_like(angle) * -50
        theta_leg[3, :] = np.ones_like(angle) * -90

        return theta_leg

    def I(self, l, m, r):

        I1 = 1 / 2 * (m * r**2)
        I2 = 1 / 12 * (m * (3 * r**2 + l**2))

        return np.array([I2, I2, I1])

    def COP_upper(self):
        l = self.utrunk_l
        w = self.utrunk_w
        M = self.utrunk_m
        h = self.utrunk_h

        g = 9.8

        X_calc = self.Pose.Xcom_ut * 1000
        Y_calc = self.Pose.Ycom_ut * 1000

        Fn = g * M * np.ones_like(X_calc)

        return X_calc, Y_calc, Fn

    def COP_lower(self):

        # lower trunk parameters
        l = self.ltrunk_l
        w = self.ltrunk_w
        M = self.ltrunk_m
        h = self.ltrunk_h

        r = 0.0914 - h / 2

        # dx = r * rot
        # alpha = np.arctan(dx / ((l + 0.005 + self.utrunk_l) / 2))

        g = 9.81

        X_calc = self.Pose.Xcom_lt * 1000
        Y_calc = self.Pose.Ycom_lt * 1000

        # plt.plot(X_calc.T)
        # plt.show()
        # exit()

        Fn = g * M * np.ones_like(X_calc)

        # plt.plot(Fn.T)
        # plt.show()
        # exit()

        return X_calc, Y_calc, Fn

    def calc_COP(self):
        # self.Pose.plotCOM()

        X_up, Y_up, Fn_up = self.COP_upper()
        X_low, Y_low, Fn_low = self.COP_lower()

        X_h = self.Pose.Xcom_h * 1000
        Y_h = self.Pose.Ycom_h * 1000
        Fn_h = np.ones_like(X_h) * self.M_head

        # reshaping to make sure all dimenesions are [1,n]
        # self.Xcalc = -X_up
        # self.Ycalc = Y_up
        self.Xcalc = (np.multiply(X_low, Fn_low) + np.multiply(X_up, Fn_up) + np.multiply(X_h, Fn_h)) / (
            Fn_up + Fn_low + Fn_h
        )
        self.Ycalc = (np.multiply(Y_low, Fn_low) + np.multiply(Y_up, Fn_up) + np.multiply(Y_h, Fn_h)) / (
            Fn_up + Fn_low + Fn_h
        )

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
        # n = len(self.robot_t)
        n = len(self.Pose.Xcom_lt)
        t = np.linspace(0, (n - 1) / 60, num=n)
        # np.asarray(self.robot_t).reshape(n, 1)

        X, Y = self.COP.Xfilt, self.COP.Yfilt
        t_c = np.linspace(0, (len(X) - 1) / self.rate, num=len(X)) + 1.1

        # print(np.max(self.Xcalc) - np.min(self.Xcalc))
        # print(np.max(self.Ycalc) - np.min(self.Ycalc))

        Xcalc = self.Xcalc
        Ycalc = self.Ycalc

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
