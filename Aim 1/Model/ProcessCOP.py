import numpy as np
import math
from scipy import signal
from scipy import stats
import pandas as pd
from numpy import radians as radians
import matplotlib.pyplot as plt


class processCOP:
    # mass=[2X1], length=[2x1], I=[2x3], theta=[4,t]
    def __init__(self, file, rate=60, fcut=5):

        cop = pd.read_csv(
            file,
            header=8,
            parse_dates=True,
            names=["UL_raw", "UR_raw", "LL_raw", "LR_raw", "X_scaled", "Y_scaled", "Reaction"],
        )

        # For cases where "offset" values are included in COP file
        if cop.iloc[0, 0] == "UL_raw":
            cop = cop.iloc[1:-1, :]
            cop = cop.astype(float)

        self.file = file

        self.Xraw = cop.X_scaled
        self.Yraw = cop.Y_scaled
        self.Rraw = cop.Reaction

        self.mat_raw = np.array([cop.UL_raw, cop.UR_raw, cop.LL_raw, cop.LR_raw])

        order = 3

        b, a = signal.butter(order, fcut, fs=rate)

        self.Xfilt = signal.filtfilt(b, a, self.Xraw)
        self.Yfilt = signal.filtfilt(b, a, self.Yraw)
        self.Rfilt = signal.filtfilt(b, a, self.Rraw)

        self.cop = cop

        self.order = order
        self.fcut = fcut
        self.rate = rate

    def switch_load_cells(self, ul=1, ur=2, ll=3, lr=4):
        file = self.file

        Tare = pd.read_csv(file, parse_dates=True, header=2, nrows=1)
        Scale = pd.read_csv(file, parse_dates=True, header=5, nrows=1)

        tare = Tare.to_numpy()[0, 0:4]
        scale = Scale.to_numpy()[0, 0:4]

        load_raw = self.mat_raw

        UL_raw = load_raw[ul - 1, :]
        UR_raw = load_raw[ur - 1, :]
        LL_raw = load_raw[ll - 1, :]
        LR_raw = load_raw[lr - 1, :]

        UL_tare = tare[ul - 1]
        UR_tare = tare[ur - 1]
        LL_tare = tare[ll - 1]
        LR_tare = tare[lr - 1]

        UL_scale = scale[ul - 1]
        UR_scale = scale[ur - 1]
        LL_scale = scale[ll - 1]
        LR_scale = scale[lr - 1]

        refwt = 1.055
        xreflength = 563.9562
        yreflength = 560.7812

        UL = refwt * (UL_raw - UL_tare) / (UL_scale - UL_tare)
        UR = refwt * (UR_raw - UR_tare) / (UR_scale - UR_tare)
        LL = refwt * (LL_raw - LL_tare) / (LL_scale - LL_tare)
        LR = refwt * (LR_raw - LR_tare) / (LR_scale - LR_tare)

        UL_o = refwt * (0 - UL_tare) / (UL_scale - UL_tare)
        UR_o = refwt * (0 - UR_tare) / (UR_scale - UR_tare)
        LL_o = refwt * (0 - LL_tare) / (LL_scale - LL_tare)
        LR_o = refwt * (0 - LR_tare) / (LR_scale - LR_tare)

        sum = UL + UR + LL + LR
        plate_wt = UL_o + UR_o + LL_o + LR_o

        x_manual = 0.5 * xreflength * ((UR - UR_o + LR - LR_o) - (UL - UL_o + LL - LL_o)) / (sum - plate_wt)
        y_manual = 0.5 * yreflength * ((UL - UL_o + UR - UR_o) - (LL - LL_o + LR - LR_o)) / (sum - plate_wt)

        b, a = signal.butter(self.order, self.fcut, fs=self.rate)

        X = signal.filtfilt(b, a, x_manual)
        Y = signal.filtfilt(b, a, y_manual)

        n = len(self.Xfilt)

        # print(np.shape(x_manual))

        self.Xfilt = X.reshape(n)
        self.Yfilt = Y.reshape(n)

        # # print(np.shape(xmanual))
        # self.Xraw = x_manual
        # self.Yraw = cop.Y_scaled

        return x_manual, y_manual

    def save_switch(self, ul=1, ur=2, ll=3, lr=4):
        xswitch, yswitch = self.switch_load_cells(ul=ul, ur=ur, ll=ll, lr=lr)
        data = self.cop

        data.X_scaled = xswitch.ravel()
        data.Y_scaled = yswitch.ravel()

        full_name = self.file[0:-4] + "_mod.csv"
        df = pd.DataFrame(data)
        df.to_csv(full_name, index=False)

        print("Saved As:" + full_name)


# file = r"C:\Users\franc\Box\Rehab Robotics Lab\Projects\PANDA Gym (# 834084)\Data\Trials\Aim III\833180_212\10-26-2022\Mat\2022_10_26_833180_212_mat_session4_toy_at_arms - original.csv"
# cc = processCOP(file)
# cc.save_switch(ul=2, ur=1)
