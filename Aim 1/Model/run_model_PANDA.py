import sys
import glob
import scipy
import pandas as pd
import numpy as np
import matplotlib as plt
from CompareCOP import compareCOP
from COP_3D import calculate_COP

# sys.path.insert(0, "../PANDA-Data-Processing")
sys.path.insert(0, r"C:\Users\franc\Documents\GitHub\PANDA-Data-Processing")
from Cmanage_PANDA import c_manage


class run_model:
    def __init__(self, aim, subID, month, day, year):
        userdirect = r"C:\Users\franc\Box\Rehab Robotics Lab\Projects\PANDA Gym (# 834084)\Data\Trials"

        self.year = year
        self.day = day
        self.month = month
        self.subID = subID
        self.aim = aim

        self.cmanage = c_manage(userdirect, aim, subID, month, day, year)

        copfile = self.get_cop()
        posefile = self.cmanage.load_3D(vidnum=4)

        # print(copfile)

        calc = calculate_COP(posefile, copfile)
        calc.calc_COP()
        self.calc = calc
        # self.comp()

    def get_cop(self):
        cop_direct = self.cmanage.subdirect + "Mat\\"

        return glob.glob(cop_direct + "\\*session2.csv")[0]

    def comp(self, start=60, stop=-1, all_metrics=False):
        # comparing COP
        Xcalc = self.calc.Xcalc.T
        Ycalc = self.calc.Ycalc.T

        Xreal = self.calc.COP.Xfilt[::2]
        Yreal = self.calc.COP.Yfilt[::2]

        Xreal, Yreal = Xreal[start:stop], Yreal[start:stop]
        Xcalc, Ycalc = Xcalc[start:stop], Ycalc[start:stop]

        win = 2

        Xcalc = scipy.ndimage.median_filter(Xcalc, win)
        Ycalc = scipy.ndimage.median_filter(Ycalc, win)

        compare = compareCOP(Xcalc, Ycalc, Xreal, Yreal, cam=1)
        comp_metrics = compare.diff_metric()
        self.compare = compare

        # print(comp_metrics)

        if all_metrics:

            path_metrics = self.calc.path_len()
            corr_metrics = self.calc.corr_COP()

            return comp_metrics, corr_metrics, path_metrics
        else:
            return comp_metrics

    def save_XY(self, folder, vidnum=4, suffix=None):

        filename = (
            "\\"
            + str(self.year)
            + "_"
            + str(self.month).zfill(2)
            + "_"
            + str(self.day).zfill(2)
            + "_833180_"
            + str(self.subID).zfill(3)
            + "_vid"
            + str(vidnum)
        )

        Title = (
            str(self.subID).zfill(3)
            + ": "
            + str(self.month).zfill(2)
            + "/"
            + str(self.day).zfill(2)
            + "/"
            + str(self.year)
        )

        save_name = folder + "\\" + filename + "_XY_" + suffix + ".png"
        self.compare.comp_XY(save=True, name=save_name, title=Title, show=False)


aim = 1
sub = 70
m = 6
d = 6
y = 2023
# sub = 21
# m = 8
# d = 16
# y = 2022
# sub = 39
# m = 9
# d = 9
# y = 2022

m = run_model(aim, sub, m, d, y)
print(m.comp(start=0))
m.compare.comp_XY(save=False, show=True)

# print(m.calc.rad_metrics())
# m.calc.plot_cop_anim()
# m.calc.optim_COP()
# m.calc.plot_cop_anim()
# comp = m.calc.load_compare()
# comp.plot_cop_anim()


# m.calc.compare_COP(start=60)
# fldr = r"C:\Users\franc\Documents\RRL\Aim 2 Data\Analysis 1"
# m.save_XY(fldr)
