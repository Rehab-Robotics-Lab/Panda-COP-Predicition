import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split


class split_data:
    def __init__(self, input, output, ids, testsize=0.25, batch_size=32):
        # Reading npy files with subject ids, pose and cop
        ids = np.load(ids)
        input = np.load(input)
        output = np.load(output)

        # splitting subject ids into train and test sets
        train, test = train_test_split(ids, test_size=testsize)

        # masks for train and test sets based on subject id
        train_bool = np.isin(ids, train)
        test_bool = np.isin(ids, test)

        # turning data into dataset and splitting into 29, 120s windows
        self.training_set = PANDA_dateset(input[train_bool], output[train_bool], ids[train_bool])
        self.test_set = PANDA_dateset(input[test_bool], output[test_bool], ids[test_bool])

        # loading datasets into dataloader
        self.train_loader = DataLoader(self.training_set, batch_size=batch_size)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size)


# Class for PANDA daset as data loader object
class PANDA_dateset(Dataset):
    def __init__(self, input, output, ids, window_size=120, num_windows=29):
        # Reading input as numpy file
        input_pose = torch.from_numpy(input)
        input_cop = torch.from_numpy(output)
        self.sub_IDs = torch.from_numpy(ids)

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

        # unrolling data for each subject into 29, 120-frame windows
        for i in range(self.tot_subjects):
            # iterating by num windows
            for j in range(num_windows):
                # updating pose
                pose[(i * num_windows) + j, :, :, :] = input_pose[
                    i, :, :, (j * num_windows) : (j * num_windows) + window_size
                ]
                # updating cop
                cop[(i * num_windows) + j, :, :] = input_cop[i, :, (j * num_windows) : (j * num_windows) + window_size]

        self.pose = pose
        self.cop = cop

        # number of subjects
        return self.num_windows * self.tot_subjects

    def __getitem__(self, idx):
        item = {"pose": self.pose[idx, :, :, :], "cop": self.cop[idx, :, :]}
        return item
