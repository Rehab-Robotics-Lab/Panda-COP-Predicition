import scipy
import seaborn as sns
from scipy import stats
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Process_model import process_model_data as prcoess_data

#Class to compare model performance
class analyze_models:
    def __init__(self,folder,split):
        self.folder=folder
        self.split=split

        self.grnd_trth=self.folder+'\\'+'ground_truth_cop_'+self.split+'.npy'
        self.info=self.folder+'\\'+'sub_info_'+self.split+'.npy'

    #Getting a compare onject based on each model
    def get_model_object(self,name):
        model_data=self.folder+'\\'+name+'_cop_'+self.split+'.npy'

        model=prcoess_data(model_data,self.grnd_trth,self.info)

        return model
    
    def plot_outliers(self,name,metric):
        model=self.get_model_object(name)
        outliers=model.metric_outliers(metric)

        print(outliers)

        for i in range(len(outliers)):
            model.plot_xy(outliers.iloc[i].ID.item(),outliers.iloc[i].Month.item())
