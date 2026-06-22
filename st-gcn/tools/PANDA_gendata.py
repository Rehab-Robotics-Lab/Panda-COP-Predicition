import os
import sys
import pickle

import scipy
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


def gendata(subject_list, aim=1):
    # Loading in data with minimum 29 120frame windows
    df = subject_list
    folder = rf"C:\Users\franc\Box\Rehab Robotics Lab\Projects\PANDA Gym (# 834084)\Personnel\Students & RAs\Francis Sowande\Aim 3"

    # extracting subject ids
    subid = df.subjectID
    subid_u = pd.unique(subid)

    # total number of subjects
    tot = len(subid)

    # start frame
    start_frame = 60

    # finals frame (total frame num + start frame offset)
    max_frame_all = np.max(subject_list.window_round) * 120

    # insatitating data arrays
    pose_data = np.zeros((tot, n_dim, n_joints, max_frame_all))
    cop_data = np.zeros((tot, 2, max_frame_all))
    model_cop = np.zeros((tot, 2, max_frame_all))
    model_cop_face = np.zeros((tot, 2, max_frame_all))
    model_cop_ears = np.zeros((tot, 2, max_frame_all))
    # subID, scale, win_num, month, group, gender
    sub_info = np.zeros((tot, 6))

    run_sum = 0

    # iterating through all ids
    for i in range(len(subid_u)):
        id = int(subid_u[i])
        print("SUBJECT ID: ", str(id).zfill(3))

        # Table of specific subject info
        subject = df.loc[df["subjectID"] == id]

        # Subect/trial date
        sub_dates_ind = pd.unique(subject.Date)
        sub_dates = pd.to_datetime(sub_dates_ind)

        indx_all = df.loc[df["subjectID"] == id].index
        # print(indx_all)

        for j in range(len(sub_dates)):
            trial_date = sub_dates[j]
            print("Date: ", str(trial_date))

            trial = subject.iloc[j].copy()

            # print("Date: ", str(trial_date))

            # Frame number infromation
            max_frames = trial.window_round * 120
            offset = trial.offset

            stop = max_frames + start_frame

            print(trial.window_round, max_frames)

            # loading object for cop predicition model (coneveniently)
            baby_best, baby_face, baby_ears = find_baby_best(aim, id, trial_date, offset, stop)

            #

            # Calc COP from best
            Xcalc_best, Ycalc_best, Xreal, Yreal = get_calc_cop(baby_best, offset, start_frame, stop)
            # Calc COP from face
            Xcalc_face, Ycalc_face, Xreal, Yreal = get_calc_cop(baby_face, offset, start_frame, stop)
            # Calc COP from ears
            Xcalc_ears, Ycalc_ears, Xreal, Yreal = get_calc_cop(baby_ears, offset, start_frame, stop)

            # Loading in trial pose
            pose = baby_best.calc.pose

            # Accouting for offset in pose data
            if offset < 0:
                Xpose, Ypose, Zpose = (
                    pose.X[start_frame - offset : stop - offset, :],
                    pose.Y[start_frame - offset : stop - offset, :],
                    pose.Z[start_frame - offset : stop - offset, :],
                )
            else:
                Xpose, Ypose, Zpose = (
                    pose.X[start_frame:stop, :],
                    pose.Y[start_frame:stop, :],
                    pose.Z[start_frame:stop, :],
                )

            # Converting classification to number
            if trial.group == "TD":
                group = 1
            elif trial.group == "BI":
                group = -1

            # Converting gender to number
            if trial.Gender == "F":
                gender = 1
            elif trial.Gender == "M":
                gender = -1

            indx = indx_all[j]

            ## subID, scale, win_num, month, group, gender
            # Filling in subject ID matrix
            sub_info[indx, 0] = id
            # Filling in scale
            sub_info[indx, 1] = trial.scale
            # Filling in number of windows
            sub_info[indx, 2] = trial.window_round
            # Filling in month
            sub_info[indx, 3] = trial.month
            # Filling in group
            sub_info[indx, 4] = group
            # Filling in gender
            sub_info[indx, 5] = gender

            # Filling in pose data matrix
            pose_data[indx, 0, :, 0:max_frames] = Xpose.T
            pose_data[indx, 1, :, 0:max_frames] = Ypose.T
            pose_data[indx, 2, :, 0:max_frames] = Zpose.T

            # Filling in cop data matrix
            cop_data[indx, 0, 0:max_frames] = Xreal.T
            cop_data[indx, 1, 0:max_frames] = Yreal.T

            # Filling in model cop data matrix
            model_cop[indx, 0, 0:max_frames] = Xcalc_best.T
            model_cop[indx, 1, 0:max_frames] = Ycalc_best.T

            # Filling in model cop data matrix for face
            model_cop_face[indx, 0, 0:max_frames] = Xcalc_face.T
            model_cop_face[indx, 1, 0:max_frames] = Ycalc_face.T

            # Filling in model cop data matrix for ears
            model_cop_ears[indx, 0, 0:max_frames] = Xcalc_ears.T
            model_cop_ears[indx, 1, 0:max_frames] = Ycalc_ears.T

            run_sum += max_frames

    # np.save(folder + "\\input_data_val.npy", pose_data)
    # np.save(folder + "\\output_data_val.npy", cop_data)
    # np.save(folder + "\\model_cop_val.npy", model_cop)
    np.save(folder + "\\model_cop_face_test.npy", model_cop_face)
    np.save(folder + "\\model_cop_ears_test.npy", model_cop_ears)
    # np.save(folder + "\\sub_info_val.npy", sub_info)

    print(run_sum)


def get_calc_cop(baby, offset, start_frame, stop):
    # comparing COP
    Xcalc = baby.calc.Xcalc.T
    Ycalc = baby.calc.Ycalc.T

    Xreal = baby.calc.COP.Xfilt[::2]
    Yreal = baby.calc.COP.Yfilt[::2]

    if len(Xreal) == 0:
        Xreal = baby.COP.Xfilt[::2]
    if len(Yreal) == 0:
        Yreal = baby.COP.Yfilt[::2]

    # Accounting for offset in cop data
    if offset == 0:
        Xreal, Yreal = Xreal[start_frame:stop], Yreal[start_frame:stop]
        Xcalc, Ycalc = Xcalc[start_frame:stop], Ycalc[start_frame:stop]
    elif offset > 0:
        Xreal, Yreal = Xreal[start_frame + offset : stop + offset], Yreal[start_frame + offset : stop + offset]
        Xcalc, Ycalc = Xcalc[start_frame:stop], Ycalc[start_frame:stop]
    elif offset < 0:
        Xcalc, Ycalc = Xcalc[start_frame - offset : stop - offset], Ycalc[start_frame - offset : stop - offset]
        Xreal, Yreal = Xreal[start_frame:stop], Yreal[start_frame:stop]

    win = 2

    Xcalc_filt = scipy.ndimage.median_filter(Xcalc, win)
    Ycalc_filt = scipy.ndimage.median_filter(Ycalc, win)

    return Xcalc_filt, Ycalc_filt, Xreal, Yreal


def find_baby_best(aim, id, trial_date, offset, stop, seg=2):
    baby_ears = run_model(aim, id, trial_date.month, trial_date.day, trial_date.year, head="ears", segment=seg)
    baby_ears.calc.calc_COP()
    metrics_ears = baby_ears.comp(offset=offset, stop=stop)

    baby_face = run_model(aim, id, trial_date.month, trial_date.day, trial_date.year, head="face", segment=seg)
    baby_face.calc.calc_COP()
    metrics_face = baby_face.comp(offset=offset, stop=stop)

    corr_ears = metrics_ears.loc[0, "Mean Corr"]
    corr_face = metrics_face.loc[0, "Mean Corr"]

    if corr_ears > corr_face:
        baby_best = baby_ears
        print("Baby best = Ears")
    elif corr_ears < corr_face:
        baby_best = baby_face
        print("Baby best = Face")

    return baby_best, baby_face, baby_ears


if __name__ == "__main__":
    aim = 1
    listfile = rf"C:\Users\franc\Box\Rehab Robotics Lab\Projects\PANDA Gym (# 834084)\Personnel\Students & RAs\Francis Sowande\Aim 3\Inclusion List Aim {aim}.xlsx"

    parser = argparse.ArgumentParser(description="PANDA Data Converter.")
    parser.add_argument("--list_file", default=listfile)
    # parser.add_argument("--out_folder", default="data/NTU-RGB-D")

    arg = parser.parse_args()

    df = pd.read_excel(listfile)

    # out_path = os.path.join(arg.out_folder, b)
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)
    gendata(df, aim=aim)
