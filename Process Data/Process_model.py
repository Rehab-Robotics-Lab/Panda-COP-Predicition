import scipy
import seaborn as sns
from scipy import stats
from scipy.signal import detrend
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import EntropyHub as EH

class process_model_data:
    def __init__(self,model_data,grnd_truth_data,sub_info,exclude_list=[],show_include=True):
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
        table=self.iterate_id()

        if len(exclude_list)>0:
            if show_include:
                table = table[~pd.MultiIndex.from_frame(table[['ID', 'Month']]).isin(exclude_list)]
                print("Showing table WITHOUT specifed exlections, New Length: ",len(table))
            else:
                table = table[pd.MultiIndex.from_frame(table[['ID', 'Month']]).isin(exclude_list)]
                print("Showing table WITH ONLY specifed exlections, New Length: ",len(table))

        #table with all varibales (included or excluded)
        self.table=table

        #saving list of excluded variables
        self.exclude_list=exclude_list
        self.show_include=show_include


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

    def gen_panda_metrics(self,id=0,month=None):
        xreal,yreal,xmodel,ymodel,_=self.subject_data(id,month)

        mtr=generate_metric([xmodel,ymodel])
        metric_table=mtr.metrics()

        _,_,_,_,sub_info=self.subject_data(id,month)

        metric_table["ID"] = id
        metric_table["Month"] = sub_info.month.item()

        return metric_table
    
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
    
    def iterate_panda_metrics(self):
        #Empty datafraem
        df = pd.DataFrame()

        #Iterate through IDs
        for i in range(len(self.ids)):
            print("--ID: ",self.ids[i]," Month: ",self.months[i])
            metrics=self.gen_panda_metrics(self.ids[i],self.months[i])
            # Appending new data to table
            df=pd.concat([df,metrics])

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



class generate_metric:
    def __init__(self,cop,win=5,order=2):
        self.x=cop[0]
        self.y=cop[1]

        dt = 1 / 30

        self.dx = scipy.signal.savgol_filter(self.x, win, order, deriv=1, delta=dt)
        self.dy = scipy.signal.savgol_filter(self.y, win, order, deriv=1, delta=dt)

        self.n=len(self.x)
    
    #Return X and Y pos. or vel. depending on specified derivative
    def get_xy(self,deriv=0):
        if deriv==0:
            return self.x, self.y
        elif deriv==1:
            return self.dx,self.dy

    # COP standard deviation
    def cop_std(self,deriv=0):
        x,y=self.get_xy(deriv)

        return np.std(x), np.std(y)
    
    # COP Root means squared
    def cop_rms(self,deriv=0):
        x,y=self.get_xy(deriv)
        #Finding sum of distances
        distances = np.sum(x**2 + y**2)
    
        return np.sqrt(distances /self.n)

    # COP Excursion 
    def cop_excusrion(self,deriv=0):
        x,y=self.get_xy(deriv)

        return (np.max(x)-np.min(x)),  (np.max(y)-np.min(y))

    #COP Entropy
    def cop_entropy(self,deriv=0,dim=2,r=0.2):
        x,y=self.get_xy(deriv)

        x_proc = detrend(x)
        x_norm = (x_proc - np.mean(x_proc)) / np.std(x_proc)
        
        y_proc = detrend(y)
        y_norm = (y_proc - np.mean(y_proc)) / np.std(y_proc)

        Ex,_, _ = EH.SampEn(detrend(x_norm), m = dim)
        Ey,_, _ = EH.SampEn(detrend(y_norm), m = dim)

        return Ex[-1],Ey[-1]

    #Mean COP
    def cop_mean(self,deriv=0): 
        x,y=self.get_xy(deriv)

        return np.mean(x), np.mean(y)
    
    # Mean Median
    def cop_median(self): 
        x,y=self.x,self.y

        return np.median(x), np.median(y)

    #Average path length
    def path_len(self): 
        x,y=self.x,self.y
    
        # Calculate Euclidean distance for each step and sum them up
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        return (np.sum(distances)/self.n)

    # COP Area
    def cop_area(self,n_std=2):
        x,y=self.x,self.y

        cov = np.cov(x, y)

        area = np.pi * (n_std ** 2) * np.sqrt(np.linalg.det(cov))
        return area

    
    def metrics(self):
        # Standard Deviation (pos. and vel.)
        stdx,stdy=self.cop_std(deriv=0)
        stdx_v,stdy_v=self.cop_std(deriv=1)

        # RMS (pos. and vel.)
        rms=self.cop_rms(deriv=0)
        rms_v=self.cop_rms(deriv=1)

        # Path length and area
        area=self.cop_area()
        path_len=self.path_len()

        # Excursion (pos. and vel.)
        exrx,exry=self.cop_excusrion(deriv=0)
        exrx_v,exry_v=self.cop_excusrion(deriv=1)

        # Entropy (pos. and vel.)
        entx,enty=self.cop_entropy(deriv=0)
        entx_v,enty_v=self.cop_entropy(deriv=1)

        # Mean (pos. and vel.)
        meanx,meany=self.cop_mean(deriv=0)
        meanx_v,meany_v=self.cop_mean(deriv=1)

        # Median (pos.)
        medx,medy=self.cop_median()

        cols=['COP Std X','COP Std Y','COP Std vX','COP Std vY', 'COP excursion X','COP excursion Y','COP excursion vX','COP excursion vY',
              'COP entropy X','COP entropy Y','COP entropy vX','COP entropy vY','COPMeanX','COPMeanY','COP mean vX','COP mean vY' ,
              'COPMedianX', 'COPMedianY','COP RMS','COP vRMS', 'COP Path Length', 'COP area']

        df=pd.DataFrame([[stdx,stdy,stdx_v,stdy_v,exrx,exry,exrx_v,exry_v,
                         entx,enty,entx_v,enty_v,meanx,meany,meanx_v,meany_v,
                         medx,medy,rms,rms_v,path_len,area]],columns=cols)
        
        return df
        