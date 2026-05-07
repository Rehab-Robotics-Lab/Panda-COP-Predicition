import os
import sys
import pickle

import argparse
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

sys.path.insert(0, r"C:\Users\franc\Documents\GitHub\Panda-COP-Predicition\Aim 1\Model")
from run_model_PANDA import run_model
from ProcessPose_3D import processpose

# training_subjects = [
#     1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
# ]
# training_cameras = [2, 3]
# max_frame = 300
toolbar_width = 30

n_people = 1
n_joints = 18
n_dim = 3


def gendata(subject_list):
    # Loading in data with minimum 29 120frame windows
    df = subject_list.loc[subject_list["window_round"] >= 29]
    folder = r"C:\Users\franc\Box\Rehab Robotics Lab\Projects\PANDA Gym (# 834084)\Personnel\Students & RAs\Francis Sowande\Aim 3"

    # extracting subject ids
    subid = df.subjectID
    subid_u = pd.unique(subid)

    # total number of subjects
    tot = len(subid_u)

    # subject aim
    aim = 1

    # start frame
    start_frame = 60
    # finals frame (total frame num + start frame offset)
    max_frame = (29 * 120) + 60

    # insatitating data arrays
    input_data = np.zeros((tot, n_dim, n_joints, max_frame - 60))
    cop_data = np.zeros((tot, 2, max_frame - 60))
    sub_id = np.zeros((tot))

    # iterating through all ids
    for i in range(tot):
        id = int(subid_u[i])
        print("SUBJECT ID: ", str(id).zfill(3))

        # Table of specific subject info
        subject = df.loc[df["subjectID"] == id]

        # Subect/trial date
        sub_dates_ind = pd.unique(subject.Date)
        sub_dates = pd.to_datetime(sub_dates_ind)
        trial_date = sub_dates[0]
        # print("Date: ", str(trial_date))

        # loading object for cop predicition model (coneveniently)
        baby = run_model(aim, id, trial_date.month, trial_date.day, trial_date.year)

        # Loading in trial pose
        pose = processpose(baby.posefile)
        Xpose, Ypose, Zpose = (
            pose.X[start_frame:max_frame, :],
            pose.Y[start_frame:max_frame, :],
            pose.Z[start_frame:max_frame, :],
        )
        # Loading in trial COP
        Xcop = baby.calc.COP.Xfilt[::2][start_frame:max_frame]
        Ycop = baby.calc.COP.Yfilt[::2][start_frame:max_frame]

        # Filling in subject ID matrix
        sub_id[i] = id

        # Filling in pose data matrix
        input_data[i, 0, :, :] = Xpose.T
        input_data[i, 1, :, :] = Ypose.T
        input_data[i, 2, :, :] = Zpose.T

        # Filling in cop data matrix
        cop_data[i, 0, :] = Xcop.T
        cop_data[i, 1, :] = Ycop.T

    np.save(folder + "\\input_data.npy", input_data)
    np.save(folder + "\\output_data.npy", cop_data)
    np.save(folder + "\\sub_id.npy", sub_id)


if __name__ == "__main__":
    listfile = r"C:\Users\franc\Box\Rehab Robotics Lab\Projects\PANDA Gym (# 834084)\Personnel\Students & RAs\Francis Sowande\Aim 2\Inclusion List.xlsx"

    parser = argparse.ArgumentParser(description="PANDA Data Converter.")
    parser.add_argument("--list_file", default=listfile)
    # parser.add_argument("--out_folder", default="data/NTU-RGB-D")

    arg = parser.parse_args()

    df = pd.read_excel(listfile)

    # out_path = os.path.join(arg.out_folder, b)
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)
    gendata(df)
