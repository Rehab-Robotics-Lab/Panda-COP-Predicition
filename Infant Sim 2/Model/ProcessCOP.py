import numpy as np
import math
from scipy import signal
import pandas as pd
from numpy import radians as radians
import matplotlib.pyplot as plt


class processCOP:
    # mass=[2X1], length=[2x1], I=[2x3], theta=[4,t]
    def __init__(self, file, rate):

        cop = pd.read_csv(
            file,
            header=8,
            parse_dates=True,
            names=["UL_raw", "UR_raw", "LL_raw", "LR_raw", "X_scaled", "Y_scaled", "Reaction"],
        )

        self.Xraw = cop.X_scaled
        self.Yraw = cop.Y_scaled
        self.Rraw = cop.Reaction

        order = 3
        fcut = 5

        b, a = signal.butter(order, fcut, fs=rate)

        self.Xfilt = signal.filtfilt(b, a, self.Xraw)
        self.Yfilt = signal.filtfilt(b, a, self.Yraw)
        self.Rfilt = signal.filtfilt(b, a, self.Rraw)
