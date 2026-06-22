import os
import sys
import glob
import scipy
import pandas as pd
import numpy as np
import matplotlib as plt
from CompareCOP import compareCOP
from COP_3D import calculate_COP
from COP_3D_3R import calculate_COP_3s

# sys.path.insert(0, "../PANDA-Data-Processing")
sys.path.insert(0, r"C:\Users\franc\Documents\GitHub\PANDA-Data-Processing")
from Cmanage_PANDA import c_manage


class run_model:
    def __init__(self, aim, subID, month, day, year, segment=2, head="face", cond="None"):
        userdirect = r"C:\Users\franc\Box\Rehab Robotics Lab\Projects\PANDA Gym (# 834084)\Data\Trials"

        self.year = year
        self.day = day
        self.month = month
        self.subID = subID
        self.aim = aim

        self.cmanage = c_manage(userdirect, aim, subID, month, day, year)

        vnum = self.vnum_from_cond(cond)

        copfile = self.get_cop(cond)
        posefile = self.cmanage.load_3D(vidnum=vnum)
        # print(posefile)
        # quit()
        self.posefile = posefile

        # print(copfile)
        if segment == 2:
            calc = calculate_COP(posefile, copfile, head=head)
        elif segment == 3:
            calc = calculate_COP_3s(posefile, copfile, head=head)

        calc.calc_COP()
        self.calc = calc
        self.vnum = vnum

        p1 = self.calc.pose.pose_idx(1)
        p2 = (self.calc.pose.pose_idx(8) + self.calc.pose.pose_idx(11)) / 2

        L = np.linalg.norm(p1 - p2, axis=0)
        self.scale = np.mean(L)
        # self.comp()

    def vnum_from_cond(self, condition):
        if condition == "None":
            vnum = 4
        else:
            cond_file = r"C:\Users\franc\Box\Rehab Robotics Lab\Projects\PANDA Gym (# 834084)\Data\Trials\Camera Calibration\Video Number List.xlsx"
            df = pd.read_excel(cond_file)
            df["Date"] = pd.to_datetime(df["Date"], format="mixed")
            subdate = pd.to_datetime(str(self.year) + "-" + str(self.month) + "-" + str(self.day))

            vnum = df.loc[(df["subjectID"] == self.subID) & ((df["Date"]) == subdate), condition].item()

        print(vnum)
        return vnum

    def get_cop(self, cond):
        cop_direct = self.cmanage.subdirect + "Mat\\"

        if cond == "None":
            ends = ["session2.csv"]
        elif cond == "Arms":
            ends = ["arms.csv", "arm.csv", "hand.csv", "hands.csv"]
        elif cond == "Feet":
            ends = ["feet.csv", "legs.csv"]

        for entry in os.scandir(cop_direct):
            if entry.is_file():  # check if it's a file
                for ed in ends:
                    if entry.name.endswith(ed):
                        copfile = entry.name

        # print(cop_direct + "\\" + copfile)

        return cop_direct + "\\" + copfile

    def comp(self, start=60, stop=-1, offset=0):
        # comparing COP
        Xcalc = self.calc.Xcalc.T
        Ycalc = self.calc.Ycalc.T

        Xreal = self.calc.COP.Xfilt[::2]
        Yreal = self.calc.COP.Yfilt[::2]

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

        compare = compareCOP(Xcalc, Ycalc, Xreal, Yreal, cam=1)
        comp_metrics = compare.diff_metric()
        self.compare = compare

        self.nframes = compare.n

        # print(comp_metrics)

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


# aim = 3
# sub = 258
# m = 8
# d = 21
# y = 2024

# m = run_model(aim, sub, m, d, y, segment=2)
# print(m.scale)
# # m.calc.compare_COP(start=60, offset=200)

# print(m.comp(start=60, offset=200))
# m.compare.comp_XY()


# m.cmanage.overlay_reproj(4, load=True, compare=False, exclude=[1, 2, 3])

# print(m.calc.rad_metrics())
# m.calc.plot_cop_anim()
# m.calc.optim_COP()
# m.calc.plot_cop_anim()
# comp = m.calc.load_compare()
# comp.plot_cop_anim()


# m.calc.compare_COP(start=60)
# fldr = r"C:\Users\franc\Documents\RRL\Aim 2 Data\Analysis 1"
# m.save_XY(fldr)
