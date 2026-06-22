import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader


class evaluate_model:
    def __init__(self, model, data, num_seq, seq_len):

        pose = data["pose"]
        scale = data["scale"]
        cop = data["cop"]
        phys_model_cop = data["model cop"]

        x_model = np.zeros((num_seq * seq_len))
        y_model = np.zeros((num_seq * seq_len))
        x_real = np.zeros((num_seq * seq_len))
        y_real = np.zeros((num_seq * seq_len))
        x_phys = np.zeros((num_seq * seq_len))
        y_phys = np.zeros((num_seq * seq_len))

        if torch.cuda.is_available():
            cop, pose, scale = cop.cuda(), pose.cuda(), scale.cuda()
            model.cuda()
        else:
            model.cpu()

        pose_scale = (torch.flatten((pose.T / scale).T, start_dim=1, end_dim=2)).permute(0, 2, 1)
        output = model(pose_scale)
        output = (output[0].permute(0, 2, 1).T * scale).T

        output_zero = ((output.T - output[:, :, 0].T).T).cpu().detach().numpy()

        cop_zero = ((cop.T - cop[:, :, 0].T).T).cpu().detach().numpy()
        phys_cop_zero = ((cop.T - cop[:, :, 0].T).T).cpu().detach().numpy()
        # cop_zero=cop.cpu().detach().numpy()

        xlast_model, ylast_model = 0, 0
        xlast_real, ylast_real = 0, 0
        xlast_phys, ylast_phys = 0, 0

        for i in range(num_seq):
            # print(len(x_model[(i*num_seq):(i*num_seq)+seq_len]),len(output_zero[i,0,:]+xlast))
            x_model[(i * seq_len) : (i * seq_len) + seq_len] = output_zero[i, 0, :] + xlast_model
            y_model[(i * seq_len) : (i * seq_len) + seq_len] = output_zero[i, 1, :] + ylast_model

            x_real[(i * seq_len) : (i * seq_len) + seq_len] = cop_zero[i, 0, :] + xlast_real
            y_real[(i * seq_len) : (i * seq_len) + seq_len] = cop_zero[i, 1, :] + ylast_real

            x_phys[(i * seq_len) : (i * seq_len) + seq_len] = phys_cop_zero[i, 0, :] + xlast_real
            y_phys[(i * seq_len) : (i * seq_len) + seq_len] = phys_cop_zero[i, 1, :] + ylast_real

            xlast_model, ylast_model = (output_zero[i, 0, -1] + xlast_model), (output_zero[i, 1, -1] + ylast_model)
            xlast_real, ylast_real = (np.mean(cop_zero[i, 0, -10:-1] + xlast_real)), np.mean(
                (cop_zero[i, 1, -10:-1] + ylast_real)
            )
            xlast_phys, ylast_phys = (np.mean(phys_cop_zero[i, 0, -1] + xlast_phys)), np.mean(
                (phys_cop_zero[i, 1, -1] + ylast_phys)
            )

        self.xmodel = np.asarray(x_model).flatten()
        self.ymodel = np.asarray(y_model).flatten()
        self.xreal = np.asarray(x_real).flatten()
        self.yreal = np.asarray(y_real).flatten()
        self.xphys = np.asarray(x_phys).flatten()
        self.yphys = np.asarray(y_phys).flatten()

        # print(self.xmodel.shape)
        MAE = self.mae_fun()
        print("MAE: ", MAE)
        print("Pearson corr: ", self.pearson_corr())
        print("Spearman corr: ", self.spearmanr_corr())

        self.plot_xy()

    def pick_model(self, model):
        if model == "data":
            xmodel, ymodel = self.xmodel, self.ymodel
        elif model == "phys":
            xmodel, ymodel = self.xphys, self.yphys

        return xmodel, ymodel

    def mse_fun(self, model):
        xmodel, ymodel = self.pick_model(model)
        xreal, yreal = self.xreal, self.yreal

        val_x = np.mean((xmodel - xreal) ** 2)
        val_y = np.mean((ymodel - yreal) ** 2)
        return (val_x + val_y) / 2

    def mae_fun(self, model):
        xmodel, ymodel = self.pick_model(model)
        xreal, yreal = self.xreal, self.yreal

        val_x = np.mean(np.abs(xmodel - xreal))
        val_y = np.mean(np.abs(ymodel - yreal))

        return (val_x + val_y) / 2

    def pearson_corr(self, model):
        xmodel, ymodel = self.pick_model(model)
        xreal, yreal = self.xreal, self.yreal

        val_x, p = stats.pearsonr(xmodel, xreal)
        val_y, p = stats.pearsonr(ymodel, yreal)

        # print("Pearson: ",val_x,val_y)

        return (val_x + val_y) / 2

    def spearmanr_corr(self, model):
        xmodel, ymodel = self.pick_model(model)
        xreal, yreal = self.xreal, self.yreal

        val_x, p = stats.spearmanr(xmodel, xreal)
        val_y, p = stats.spearmanr(ymodel, yreal)

        # print("Spearman: ",val_x,val_y)

        return (val_x + val_y) / 2

    def plot_xy(self, model):
        xmodel, ymodel = self.pick_model(model)
        xreal, yreal = self.xreal, self.yreal

        n = len(xmodel)
        t = (np.linspace(0, n / 30, num=n)).reshape(n, 1)

        plt.clf()

        # fig, ax = plt.subplots(2, 1)
        color_real = np.array([10, 35, 175]) / 255
        color_calc = np.array([255, 0, 30]) / 255

        plt.subplot(2, 1, 1)
        plt.plot(t, xreal, color=color_real)
        plt.plot(t, xmodel, color=color_calc)
        plt.legend(["Grnd Trth", "Calculated"])
        plt.title("X COP")
        plt.xlabel("Time (s)")
        plt.ylabel("COP X (mm)")
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(t, yreal, color=color_real)
        plt.plot(t, ymodel, color=color_calc)
        plt.legend(["Grnd Trth", "Calculated"])
        plt.title("Y COP")
        plt.xlabel("Time (s)")
        plt.ylabel("COP Y (mm)")
        plt.grid()

        plt.tight_layout()

        plt.show()


train_data = loader.train_loader
test_data = loader.test_loader


def check_model(loader, model, split="Test"):
    if split == "Test":
        sett = loader.training_set
        load = loader.train_loader
    elif split == "Train":
        sett = loader.test_set
        load = loader.test_loader
    id = np.linspace(0, len(sett.sub_IDs) - 1, len(sett.sub_IDs))
    print(np.random.choice(id, 1))
    dat = sett.get_data_reroll(np.random.choice(id, 1))
    reroll = compare_cop_reroll(model, dat)


modd = loaded_model
check_model(loader, modd, split="Test")
