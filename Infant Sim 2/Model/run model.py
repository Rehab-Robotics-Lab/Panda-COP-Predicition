from Inv_dynamics import inv_dynamics
from ProcessCOP import processCOP
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# function converts angles from infant simulator to angles for RNE/FK
def sim_to_theta(angle):
    # array of all angles
    theta_larm = np.zeros((4, len(angle)))

    # theta 2 is angles from robot and theta 4 is fixed elbow angle of about 30 deg
    theta_larm[1, :] = angle
    theta_larm[3, :] = np.ones_like(angle) * 70

    return theta_larm


def I(l, m, r):
    I1 = 1 / 2 * (m * r**2)
    I2 = 1 / 12 * (m * (3 * r**2 + l))
    # interia tensor of cylinder(positions of I1 and I2 depend on definition of axes)
    # b/c of DH param x axis should always be along cylinder length
    return np.array([I1, I2, I2])


# infant parameters
uarm_len = 0.1
uarm_m = 0.377
uarm_r = 0.20
uarm_I = I(uarm_len, uarm_m, uarm_r)
# forarm/lower arm length 80mm
larm_len = 0.080
larm_m = 0.3
larm_r = 0.15
larm_I = I(larm_len, larm_m, larm_r)

# upper and lower arm paramters [2x5]
arm_m = np.array([uarm_m, larm_m])
arm_len = np.array([uarm_len, larm_len])
arm_I = np.matrix([larm_I, uarm_I])


# # upper arm length 100mm
# uleg_len = 0.110
# uleg_m = 1
# uleg_r = 0.25
# uleg_I = I(uleg_len, uleg_m, uleg_r)
# # calf/lower leg length 100mm
# lleg_len = 0.10
# lleg_m = 1
# lleg_r = 0.30
# lleg_I = I(lleg_len, lleg_m, lleg_r)


# upper and lower leg paramters [2x5]

# reading in motor data
angs = pd.read_csv(r"C:\Users\franc\Documents\Infant_Sim_data\load tests\larm_101.csv")
# sim_angle = (((angs.position - np.mean(angs.position[0:10])) / 4095) * 360) - 45
sim_angle = angs.angle

load = angs.load * 1.5 / 1000

# putting angles from robot in fromat with angles from all limbs
left_arm_angles = sim_to_theta(sim_angle.to_numpy())

# time step between input angles
dt = 0.045

# getting inverse dynamics
test = inv_dynamics(left_arm_angles, arm_len, arm_m, arm_I, dt)
T, F = test.calc()

t = angs.time

# plt.plot(t, (T))
# plt.plot(t, load)
# plt.legend(["Calculated", "Grnd Trth"])
# plt.show()
# test.vis_FK()

l = 0.13
w = 0.2
M = 2.312 - 0.677

X_calc = (F * 0.5 * w) / (F + 9.81 * M)
Y_calc = ((F * l / 2) - T) / (F + 9.81 * M)

X_calc = X_calc * 1000
Y_calc = Y_calc * 1000

cop_file = r"C:\Users\franc\Documents\Infant_Sim_data\load tests\2024-10-15_0_27_Sim_10_3.csv"

rate = 60

COP = processCOP(cop_file, rate)

X, Y = COP.Xfilt, COP.Yfilt
t_c = np.linspace(0, (len(X) - 1) / rate, num=len(X))

plt.plot(t_c + t[0], X)
plt.plot(t, X_calc)
plt.legend(["Grnd Trth", "Calculated"])
plt.show()
