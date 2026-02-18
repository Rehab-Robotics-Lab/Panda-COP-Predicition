import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import scipy
from scipy.spatial.transform import Rotation as R


class infant_parameters:
    def __init__(self):
        self.bodyparts = np.array(
            [
                "head",
                "neck",
                "upp trunk",
                "low trunk",
                "upp arm",
                "low arm",
                "hand",
                "upp leg",
                "low leg",
                "foot",
            ]
        )

    def length(self):
        label = ["a0", "a1", "R2"]
        head = np.array([0.121062, 0.000419, 0.31])
        neck = np.array([0.016266, 0.000250, 0.44])
        upp_trunk = np.array([0.108916, 0.000864, 0.50])
        low_trunk = np.array([0.105405, 0.000082, 0.008])
        upp_arm = np.array([0.068132, 0.000679, 0.43])
        low_arm = np.array([0.079878, 0.000536, 0.41])
        hand = np.array([0.043363, 0.000434, 0.24])
        upp_leg = np.array([0.093829, 0.001002, 0.59])
        low_leg = np.array([0.093812, 0.000758, 0.65])
        foot = np.array([0.07264, 0.000468, 0.32])

        table = pd.DataFrame(
            [head, neck, upp_trunk, low_trunk, upp_arm, low_arm, hand, upp_leg, low_leg, foot],
            index=self.bodyparts,
            columns=label,
        )

        return table

    def radius(self):
        label = ["a0", "a1", "R2"]
        head = np.array([0.075896, 0.000208, 0.31])
        neck = np.array([0.023255, 0.000038, 0.01])
        upp_trunk = np.array([0.063507, 0.000457, 0.42])
        low_trunk = np.array([0.059488, -0.00012, 0.06])
        upp_arm = np.array([0.033343, 0.000258, 0.21])
        low_arm = np.array([0.03627, 0.000235, 0.48])
        hand = np.array([0.018992, 0.000159, 0.17])
        upp_leg = np.array([0.046501, 0.000355, 0.28])
        low_leg = np.array([0.041651, 0.000338, 0.65])
        foot = np.array([0.025654, 0.000239, 0.32])

        table = pd.DataFrame(
            [head, neck, upp_trunk, low_trunk, upp_arm, low_arm, hand, upp_leg, low_leg, foot],
            index=self.bodyparts,
            columns=label,
        )

        return table

    def mass(self):
        label = ["a0", "a1", "a2", "a3", "a4", "R2"]
        head = np.array([1.31537, 0.0182805, 0, 0, 0, 0.60])
        neck = np.array([0.0654286, 0.00131708, 0, 0, 0, 0.34])
        upp_trunk = np.array([0.923717, 0.0215418, 0, 0, 0, 0.55])
        low_trunk = np.array([1.41229, 0.00499202, 0, 0, 0, 0.07])
        upp_arm = np.array([0.100717, 0.00265699, 0, 0, 0, 0.36])
        low_arm = np.array([0.119954, 0.00121999, 0, 0, 0, 0.20])
        hand = np.array([0.0383346, 0.0007862, 0, 0, 0, 0.32])
        upp_leg = np.array([0.294021, 0.0101128, 0, 0, 0, 0.50])
        low_leg = np.array([0.160717, 0.00415153, 0, 0, 0, 0.54])
        foot = np.array([0.129924, -0.0117758, 0.000795144, -0.0000181594, 0.000000141411, 0.32])

        total = head + neck + upp_trunk + low_trunk + (upp_arm + low_arm + hand + upp_leg + low_leg + foot) * 2
        total[-1] = 0

        bodyparts = np.append(self.bodyparts, "total")

        table = pd.DataFrame(
            [head, neck, upp_trunk, low_trunk, upp_arm, low_arm, hand, upp_leg, low_leg, foot, total],
            index=bodyparts,
            columns=label,
        )

        return table

    def inertia_AP(self):
        label = ["a0", "a1", "R2"]
        head = np.array([0.0026308, 0.0000677586, 0.46])
        neck = np.array([0.0000363144, 0.000000808773, 0.18])
        upp_trunk = np.array([0.00143154, 0.000112302, 0.56])
        low_trunk = np.array([0.0041, 0, 1])
        upp_arm = np.array([0.0000253984, 0.0000049323, 0.39])
        low_arm = np.array([0.0000715475, 0.00000243797, 0.31])
        hand = np.array([0.00000473021, 0.000000453652, 0.25])
        upp_leg = np.array([0.0000273338, 0.000034732, 0.51])
        low_leg = np.array([0.0000737481, 0.0000101945, 0.61])
        foot = np.array([0.00000107022, 0.00000192868, 0.33])

        table = pd.DataFrame(
            [head, neck, upp_trunk, low_trunk, upp_arm, low_arm, hand, upp_leg, low_leg, foot],
            index=self.bodyparts,
            columns=label,
        )

        return table

    def inertia_TV(self):
        label = ["a0", "a1", "a2", "a3", "a4", "R2"]
        head = np.array([0.00284029, 0.0000798008, 0, 0, 0, 0.58])
        neck = np.array([0.0000283259, -0.00000025007, 0.0000000124971, 0, 0, 0.24])
        upp_trunk = np.array([0.000896757, 0.0000903352, 0, 0, 0, 0.52])
        low_trunk = np.array([0.0034, 0, 0, 0, 0, 1])
        upp_arm = np.array([0.0000361068, 0.00000526363, 0, 0, 0, 0.38])
        low_arm = np.array([0.0000722222, 0.00000247629, 0, 0, 0, 0.32])
        hand = np.array([0.00000489372, 0.000000511117, 0, 0, 0, 0.30])
        upp_leg = np.array([0.0000464443, 0.0000356185, 0, 0, 0, 0.53])
        low_leg = np.array([0.0000724727, 0.0000102857, 0, 0, 0, 0.62])
        foot = np.array([0.000117714, -0.0000151977, 0.000000944832, -0.0000000208734, 0.000000000160463, 0.53])

        table = pd.DataFrame(
            [head, neck, upp_trunk, low_trunk, upp_arm, low_arm, hand, upp_leg, low_leg, foot],
            index=self.bodyparts,
            columns=label,
        )

        return table

    def inertia_LO(self):
        label = ["a0", "a1", "R2"]
        head = np.array([0.00209303, 0.0000615005, 0.49])
        neck = np.array([0.0000539398, 0.000000994507, 0.13])
        upp_trunk = np.array([0.00169946, 0.0000688296, 0.46])
        low_trunk = np.array([0.00297476, 0.0000253195, 0.10])
        upp_arm = np.array([0.0000267658, 0.0000011115, 0.24])
        low_arm = np.array([0.0000310432, 0.000000358568, 0.09])
        hand = np.array([0.00000809348, 0.000000224432, 0.21])
        upp_leg = np.array([0.000185995, 0.0000106668, 0.31])
        low_leg = np.array([0.00000458451, 0.0000024279, 0.39])
        foot = np.array([0.00000589505, 0.00000117298, 0.18])

        table = pd.DataFrame(
            [head, neck, upp_trunk, low_trunk, upp_arm, low_arm, hand, upp_leg, low_leg, foot],
            index=self.bodyparts,
            columns=label,
        )

        return table

    def get_length(self, age):
        df = self.length()

        x = np.matrix([[1], [np.pow(age, 1)]])

        mat = (df.loc[:, ["a0", "a1"]]).to_numpy()

        table = pd.DataFrame(mat * x, self.bodyparts)

        return table

    def get_radius(self, age):
        df = self.radius()
        x = np.matrix([[1], [np.pow(age, 1)]])

        mat = (df.loc[:, ["a0", "a1"]]).to_numpy()

        table = pd.DataFrame(mat * x, self.bodyparts)

        return table

    def get_mass(self, age):
        df = self.mass()

        x = np.matrix([[1], [np.pow(age, 1)], [np.pow(age, 2)], [np.pow(age, 3)], [np.pow(age, 4)]])

        mat = (df.loc[:, ["a0", "a1", "a2", "a3", "a4"]]).to_numpy()

        table = pd.DataFrame(mat * x, np.append(self.bodyparts, "total"))

        # print(table)

        # # print(mat)

        # print(table[0:4].sum() + (table[4:-2].sum()) * 2)
        return table

    def get_inertia(self, age):
        I_xx = self.inertia_TV()
        I_yy = self.inertia_LO()
        I_zz = self.inertia_AP()

        x2 = np.matrix([[1], [np.pow(age, 1)]])
        x4 = np.matrix([[1], [np.pow(age, 1)], [np.pow(age, 2)], [np.pow(age, 3)], [np.pow(age, 4)]])

        Ix = (I_xx.loc[:, ["a0", "a1", "a2", "a3", "a4"]]).to_numpy() * x4
        Iy = (I_yy.loc[:, ["a0", "a1"]]).to_numpy() * x2
        Iz = (I_zz.loc[:, ["a0", "a1"]]).to_numpy() * x2

        table = pd.DataFrame(np.hstack((Ix, Iy, Iz)), self.bodyparts, ["Ix", "Iy", "Iz"])

        return table

    def age_from_mass(self, mass):

        df = self.mass()

        # print(df)

        mat = np.flip(df.loc["total", ["a0", "a1", "a2", "a3", "a4"]].to_numpy())
        # print(mat)

        mat[-1] = mat[-1] - mass
        # print(mat)

        roots = np.roots(mat)

        real_roots = roots[np.isreal(roots)]
        real_roots = np.real(real_roots[real_roots > 0])

        if np.shape(real_roots) == (1,):
            print("Root Found")
            return real_roots[0]

        else:
            print("No roots found")

    def age_from_length(self, length, bodypart):

        df = self.length()

        mat = np.flip(df.loc[bodypart, ["a0", "a1"]].to_numpy())
        # print(mat)

        mat[-1] = mat[-1] - length
        # print(mat)

        roots = np.roots(mat)

        real_roots = roots[np.isreal(roots)]
        real_roots = np.real(real_roots[real_roots > 0])

        if np.shape(real_roots) == (1,):
            print("Root Found")
            print(real_roots[0])
            return real_roots[0]

        else:
            print("No roots found")

    def guess_age(self, mass, lengths):
        up_arm, low_arm, up_leg, low_leg = lengths[0], lengths[1], lengths[2], lengths[3]

        mass_age = self.age_from_mass(mass)

        uarm_age = self.age_from_length(up_arm, "upp arm")
        larm_age = self.age_from_length(low_arm, "low arm")
        uleg_age = self.age_from_length(up_leg, "upp leg")
        lleg_age = self.age_from_length(low_leg, "low leg")

        np.mean([mass_age, uarm_age, larm_age, uleg_age, lleg_age])
