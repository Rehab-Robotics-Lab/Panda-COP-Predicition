import numpy as np
import math
from scipy import signal
import pandas as pd
from numpy import radians as radians
import matplotlib.pyplot as plt


class processCOP:
    # mass=[2X1], length=[2x1], I=[2x3], theta=[4,t]
    def __init__(self, file, rate, fcut=5):

        cop = pd.read_csv(
            file,
            header=8,
            parse_dates=True,
            names=["UL_raw", "UR_raw", "LL_raw", "LR_raw", "X_scaled", "Y_scaled", "Reaction"],
        )

        # params = pd.read_csv(file, on_bad_lines="skip")

        self.Xraw = cop.X_scaled
        self.Yraw = cop.Y_scaled
        self.Rraw = cop.Reaction

        order = 3

        b, a = signal.butter(order, fcut, fs=rate)

        self.Xfilt = signal.filtfilt(b, a, self.Xraw)
        self.Yfilt = signal.filtfilt(b, a, self.Yraw)
        self.Rfilt = signal.filtfilt(b, a, self.Rraw)

        # self.Xnorm = np.divide(self.Xfilt, self.Rfilt) * np.mean(self.Rfilt)
        # self.Ynorm = np.divide(self.Yfilt, self.Rfilt) * np.mean(self.Rfilt)
        self.cop = cop

    def switch_load_cells(self):
        cop = self.cop

        UL_raw = cop["UL_raw"]
        UR_raw = cop["UR_raw"]
        LL_raw = cop["LL_raw"]
        LR_raw = cop["LR_raw"]

        UL_tare = 4
        UR_tare = 4
        LL_tare = 4
        LR_tare = 4.05

        UL_scale = 3902.367
        UR_scale = 3709.567
        LL_scale = 3721.4
        LR_scale = 3817.883

        refwt = 1.055
        xreflength = 563.9562
        yreflength = 560.7812

        x_scale = cop["X_scaled"]
        y_scale = cop["Y_scaled"]

        y = cop["y"]
        x = cop["x"]

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
