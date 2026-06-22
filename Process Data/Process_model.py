import scipy
import seaborn as sns
from scipy import stats
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class process_model_data:
    def __init__(self,model_data,grnd_truth_data,sub_info):
        #Model COP
        self.model_cop=np.load(model_data)
        #Ground Truth COP
        self.grnd_truth_cop=np.load(grnd_truth_data)

        #Loading subject info as table
        self.info=pd.DataFrame(np.load(sub_info),columns=["ID","scale","windows","month","group","gender"])

        #All subject ids and months
        self.ids=self.info.ID
        self.months=self.info.month

        #Table with all metrics
        self.table=self.iterate_id()

    #Function to pull data for based on subject ID and month
    def subject_data(self,id,month=None):
        info=self.info

        #Finding line with subject info and index of subject
        if month==None:
            sub_info=info.loc[info.ID == id]
            indx=info.index[info.ID == id].to_list()[0]
        else:
            sub_info=info.loc[(info['ID']==id)&(info['month']==month)]
            indx=info.index[(info['ID']==id)&(info['month']==month)].to_list()[0]

        #Getting number of windows and number of frames
        num_windows=sub_info.windows.item()
        max_frames=int(120*num_windows)

        #Loading model and ground-truth data
        model_cop=self.model_cop[indx,:,0:max_frames]
        real_cop=self.grnd_truth_cop[indx,:,0:max_frames]

        #Data along X and Y axes
        Xreal,Yreal=real_cop[0,:],real_cop[1,:]
        Xmodel,Ymodel=model_cop[0,:],model_cop[1,:]

        # #Zeroing ground truth data
        # Xreal=Xreal-np.mean(Xreal[0:15])
        # Yreal=Yreal-np.mean(Yreal[0:15])

        # #Zeroing model data
        # Xmodel=Xmodel-np.mean(Xmodel[0:15])
        # Ymodel=Ymodel-np.mean(Ymodel[0:15])

        #Zeroing ground truth data
        Xreal=Xreal-np.mean(Xreal[0])
        Yreal=Yreal-np.mean(Yreal[0])

        #Zeroing model data
        Xmodel=Xmodel-np.mean(Xmodel[0])
        Ymodel=Ymodel-np.mean(Ymodel[0])

        return Xreal,Yreal,Xmodel,Ymodel,sub_info
        
    #MSE per subject
    def mse_fun(self,id,month=None):
        xreal,yreal,xmodel,ymodel,_=self.subject_data(id,month)

        val_x = np.mean((xmodel - xreal) ** 2)
        val_y = np.mean((ymodel - yreal) ** 2)
        return val_x,val_y,(val_x + val_y) / 2
    
    #MAE per subject
    def mae_fun(self,id,month=None):
        xreal,yreal,xmodel,ymodel,_=self.subject_data(id,month)

        val_x = np.mean(np.abs(xmodel - xreal))
        val_y = np.mean(np.abs(ymodel - yreal))

        return val_x,val_y,(val_x + val_y) / 2
    
    #Pearson correlation per subject
    def pearson_corr(self,id,month=None):
        xreal,yreal,xmodel,ymodel,_=self.subject_data(id,month)

        val_x, p = stats.pearsonr(xmodel, xreal)
        val_y, p = stats.pearsonr(ymodel, yreal)

        # print("Pearson: ",val_x,val_y)

        return val_x,val_y,(val_x + val_y) / 2

    #Spearman correlation per subject
    def spearmanr_corr(self,id,month=None):
        xreal,yreal,xmodel,ymodel,_=self.subject_data(id,month)

        val_x, p = stats.spearmanr(xmodel, xreal)
        val_y, p = stats.spearmanr(ymodel, yreal)

        # print("Spearman: ",val_x,val_y)

        return val_x,val_y,(val_x + val_y) / 2

    # Table with metrics
    def metrics(self,id=0,month=None):
      maex,maey,mae_av=self.mae_fun(id,month)
      pearx,peary,pear_av=self.pearson_corr(id,month)
      spearx,speary,spear_av=self.spearmanr_corr(id,month)

      _,_,_,_,sub_info=self.subject_data(id,month)

      data1 = {
            "ID":id,
            "Month":sub_info.month.item(),
            "MAE X": maex,
            "MAE Y": maey,
            "Pears X": pearx,
            "Pears Y": peary,
            "Spear X": spearx,
            "Spear Y": speary,
            "Mean MAE": mae_av,
            "Mean Pears": pear_av,
            "Mean Spear":spear_av}

      df = pd.DataFrame([data1], index=[0])

      return df


    def plot_xy(self,id,month=None):
        xreal,yreal,xmodel,ymodel,_=self.subject_data(id,month)

        n = len(xreal)
        t = (np.linspace(0, n / 30, num=n)).reshape(n, 1)

        plt.clf()

        # fig, ax = plt.subplots(2, 1)
        color_real = np.array([10, 35, 175]) / 255
        color_calc = np.array([255, 0, 30]) / 255
        color_dat = np.array([25, 130, 60]) / 255

        plt.subplot(2, 1, 1)
        plt.plot(t, xreal, color=color_real)
        plt.plot(t, xmodel, color=color_calc)
        plt.legend(["Grnd Trth", "Model"])
        plt.title("X COP")
        plt.xlabel("Time (s)")
        plt.ylabel("COP X (mm)")
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(t, yreal, color=color_real)
        plt.plot(t, ymodel, color=color_calc)
        plt.legend(["Grnd Trth", "Model",])
        plt.title("Y COP")
        plt.xlabel("Time (s)")
        plt.ylabel("COP Y (mm)")
        plt.grid()

        plt.tight_layout()

        plt.show()
    
    def iterate_id(self):        
        #Empty datafraem
        df = pd.DataFrame()

        #Iterate through IDs
        for i in range(len(self.ids)):
            # Appending new data to table
            df=pd.concat([df,self.metrics(self.ids[i],self.months[i])])

        return df

    def metric_outliers(self,metric):
        #Table with all results
        table=self.table
        
        #Finding outliers with IQR
        Q1 = table[metric].quantile(0.25)
        Q3 = table[metric].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        #Table with all detected outliers
        outliers = table[(table[metric] < lower_bound) | (table[metric] > upper_bound)]

        return outliers







        