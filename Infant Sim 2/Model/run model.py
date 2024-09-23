from infant_sim2_model import infant_sim2
import numpy as np


# function converts angles from infant simulator to angles for RNE/FK
def sim_to_theta(angle):
    # array of all angles
    theta_larm = np.zeros((4, len(angle)))

    # theta 2 is angles from robot and theta 4 is fixed elbow angle of about 30 deg
    theta_larm[1, :] = angle
    theta_larm[3, :] = np.ones_like(angle) * 30

    return theta_larm


def I(l, m, r):
    I1 = 1 / 2 * (m * r**2)
    I2 = 1 / 12 * (m * (3 * r**2 + l))
    # interia tensor of cylinder(positions of I1 and I2 depend on definition of axes)
    # b/c of DH param x axis should always be along cylinder length
    return I1, I2, I2


# infant parameters
uarm_len = 1
uarm_m = 1
uarm_r = 0.20
uarm_Ix, uarm_Iy, uarm_Iz = I(uarm_len, uarm_m, uarm_r)
# print(I(uarm_len, uarm_m, uarm_r))

Uarm = np.array([uarm_len, uarm_m, uarm_Ix, uarm_Iy, uarm_Iz])

# forarm/lower arm length 80mm
larm_len = 0.80
larm_m = 1
larm_r = 0.20
larm_Ix, larm_Iy, larm_Iz = I(larm_len, larm_m, larm_r)
# print(I(larm_len, larm_m, larm_r))

Larm = np.array([larm_len, larm_m, larm_Ix, larm_Iy, larm_Iz])

# upper and lower arm paramters [2x5]
arms = np.matrix([Uarm, Larm])


# upper arm length 100mm
uleg_len = 0.110
uleg_m = 1
uleg_r = 0.25
uleg_Ix, uleg_Iy, uleg_Iz = I(uleg_len, uleg_m, uleg_r)

Uleg = np.array([uleg_len, uleg_m, uleg_Ix, uleg_Iy, uleg_Iz])

# calf/lower leg length 100mm
lleg_len = 0.10
lleg_m = 1
lleg_r = 0.30
lleg_Ix, lleg_Iy, lleg_Iz = I(lleg_len, lleg_m, lleg_r)

Lleg = np.array([lleg_len, lleg_m, lleg_Ix, lleg_Iy, lleg_Iz])

# upper and lower leg paramters [2x5]
legs = np.matrix([Uleg, Lleg])


left_arm = sim_to_theta(np.array([50, 40, 20, 30, 40, 50, 60]))

all_limbs = np.tile(left_arm[..., None], 4)


test = infant_sim2(all_limbs, arms, legs)
test.inv_dynamics(arms, "larm")
# test.vis_FK()
