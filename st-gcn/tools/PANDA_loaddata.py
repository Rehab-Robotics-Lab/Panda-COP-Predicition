import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy
from sklearn.model_selection import train_test_split


class load_data:
    def __init__(self, input, output, ids, model_cop, testsize=0.25, batch_size=32):
        # Reading npy files with subject ids, pose and cop
        ids = np.load(ids)
        input = self.filter_pose(np.load(input))
        self.input = input

        # loading in model cop
        output = np.load(output)
        model_cop = np.load(model_cop)

        self.tot_subjects = len(ids)
        scale = self.get_scale()

        # splitting subject ids into train and test sets
        train, test = train_test_split(ids, test_size=testsize)

        # masks for train and test sets based on subject id
        train_bool = np.isin(ids, train)
        test_bool = np.isin(ids, test)

        # turning data into dataset and splitting into 29, 120s windows
        self.training_set = PANDA_dateset(
            input[train_bool], output[train_bool], ids[train_bool], scale[train_bool], model_cop[train_bool]
        )
        self.test_set = PANDA_dateset(
            input[test_bool], output[test_bool], ids[test_bool], scale[test_bool], model_cop[train_bool]
        )

        # loading datasets into dataloader
        self.train_loader = DataLoader(self.training_set, batch_size=batch_size)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size)

    def filter_pose(self, input, win=5, order=1):
        return scipy.signal.savgol_filter(input, win, order, axis=3)

    def get_scale(self):
        pose = self.input
        neck = pose[:, :, 1, :]
        mid_hip = (pose[:, :, 8, :] + pose[:, :, 11, :]) / 2

        lengths = np.linalg.norm(neck - mid_hip, axis=1)
        scale = np.mean(lengths, axis=1)

        return scale


# Class for PANDA daset as data loader object
class PANDA_dateset(Dataset):
    def __init__(self, input, output, ids, scale, model_cop, window_size=480, num_windows=7):
        # Reading input as numpy file
        input_pose = torch.from_numpy(input)
        input_cop = torch.from_numpy(output)
        self.sub_IDs = torch.from_numpy(ids)
        trunk_scale = torch.from_numpy(scale)
        calc_cop = torch.from_numpy(model_cop)

        # datset parameters
        self.window_size = window_size
        self.num_windows = num_windows
        self.tot_subjects = len(self.sub_IDs)

        # data typer parameters
        self.num_kp = input_pose.shape[2]
        self.dim_size = input_pose.shape[1]

        # initlizing tensors for data unrolled into 120s windows
        pose = torch.zeros((self.tot_subjects * num_windows, self.dim_size, self.num_kp, self.window_size))
        cop = torch.zeros((self.tot_subjects * num_windows, 2, self.window_size))
        cop_model = torch.zeros((self.tot_subjects * num_windows, 2, self.window_size))
        scale_len = torch.zeros((self.tot_subjects * self.num_windows))

        # unrolling data for each subject into 29, 120-frame windows
        for i in range(self.tot_subjects):
            # iterating by num windows
            for j in range(num_windows):
                # updating pose
                pose[(i * num_windows) + j, :, :, :] = input_pose[
                    i, :, :, (j * window_size) : (j * window_size) + window_size
                ]
                # updating cop
                cop[(i * num_windows) + j, :, :] = input_cop[i, :, (j * window_size) : (j * window_size) + window_size]

                # updating model cop
                cop_model[(i * num_windows) + j, :, :] = calc_cop[
                    i, :, (j * window_size) : (j * window_size) + window_size
                ]

                scale_len[(i * num_windows) + j] = trunk_scale[i]

        # Saving unrolled data
        self.pose = pose
        self.cop = cop
        self.model_cop = cop_model
        self.scale_len = scale_len

        # Save rolled data
        self.pose_roll = input_pose
        self.cop_roll = input_cop
        self.model_cop_roll = calc_cop
        self.scale_len_roll = trunk_scale

        # number of subjects

    def __len__(self):
        return self.num_windows * self.tot_subjects

    def __getitem__(self, idx):
        item = {
            "pose": self.pose[idx, :, :, :],
            "cop": self.cop[idx, :, :],
            "model_cop": self.model_cop[idx, :, :],
            "scale": self.scale_len[idx],
        }
        return item

    def get_data_reroll(self, idx):
        print(idx)
        print("----------SUBJECT ID", self.sub_IDs[idx].item())
        pose = torch.zeros((self.num_windows, self.dim_size, self.num_kp, self.window_size))
        cop = torch.zeros((self.num_windows, 2, self.window_size))
        cop_model = torch.zeros((self.num_windows, 2, self.window_size))
        scale_len = torch.ones((self.num_windows))

        scale = self.scale_len_roll[idx]

        num_windows = self.num_windows
        window_size = self.window_size
        print(self.cop_roll.shape)

        for j in range(num_windows):
            # updating pose
            pose[j, :, :, :] = self.pose_roll[idx, :, :, (j * window_size) : (j * window_size) + window_size]

            # updating cop
            cop[j, :, :] = self.cop_roll[idx, :, (j * window_size) : (j * window_size) + window_size]

            # updating model cop
            cop_model[j, :, :] = self.model_cop_roll[idx, :, (j * window_size) : (j * window_size) + window_size]

            scale_len[j] = scale

        item = {"pose": pose, "cop": cop, "model_cop": cop_model, "scale": scale_len}
        return item

    def get_data_batch(self, idx):
        print(idx)
        print("----------SUBJECT ID", self.sub_IDs[idx].item())

        item = {
            "pose": self.pose[idx, :, :, :],
            "cop": self.cop[idx, :, :],
            "model_cop": self.cop_model[idx, :, :],
            "scale": self.scale_len[idx],
        }
        return item


input_file = r"C:\Users\franc\Box\Rehab Robotics Lab\Projects\PANDA Gym (# 834084)\Personnel\Students & RAs\Francis Sowande\Aim 3\input_data.npy"
output_file = r"C:\Users\franc\Box\Rehab Robotics Lab\Projects\PANDA Gym (# 834084)\Personnel\Students & RAs\Francis Sowande\Aim 3\output_data.npy"
id_file = r"C:\Users\franc\Box\Rehab Robotics Lab\Projects\PANDA Gym (# 834084)\Personnel\Students & RAs\Francis Sowande\Aim 3\sub_id.npy"

loader = load_data(input_file, output_file, id_file)
