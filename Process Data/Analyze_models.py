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
    def __init__(self,folder,split,exclude_vars=[],aim=3,show_included=True):
        self.folder=folder
        self.split=split

        self.grnd_trth=self.folder+'\\'+'ground_truth_cop_'+self.split+'.npy'
        self.info=self.folder+'\\'+'sub_info_'+self.split+'.npy'

        #Reading in table with exclusions
        exclude_table=pd.read_excel(r"C:\Users\franc\Box\Rehab Robotics Lab\Projects\PANDA Gym (# 834084)\Personnel\Students & RAs\Francis Sowande\Aim "+str(aim)+"\\Exclusion List Aim 3.xlsx")
        #Isolating rows with specifed includion
        exclude_table = exclude_table.loc[exclude_table["exclude"].isin(exclude_vars), :]
        #saving multi index Subject ID and month values
        self.exclude_list = pd.MultiIndex.from_frame(exclude_table[['subjectID', 'month']])
        
        #Varibale to determine if we view data without excluded varibale or only with excluded variables
        self.show_included=show_included


    #Getting a compare onject based on each model
    def get_model_object(self,name):
        model_data=self.folder+'\\'+name+'_cop_'+self.split+'.npy'

        model=prcoess_data(model_data,self.grnd_trth,self.info,exclude_list=self.exclude_list,show_include=self.show_included)
        

        return model
    
    def plot_outliers(self,name,metric):
        model=self.get_model_object(name)
        outliers=model.metric_outliers(metric)

        print(outliers)

        for i in range(len(outliers)):
            model.plot_xy(outliers.iloc[i].ID.item(),outliers.iloc[i].Month.item())

    def get_metric_table(self,model):
        #Reading in metric table
        name=self.folder[0:-11]+"\\Panda Metrics"+"\\"+model+"_panda_metrics.xlsx"
        print(name)
        df=pd.read_excel(name)

        #If exclusion is specidied only return part of the included/excluded results
        if len(self.exclude_list)>0:
            if self.show_included:
                df = df[~pd.MultiIndex.from_frame(df[['ID', 'Month']]).isin(self.exclude_list)]
            else:
                df = df[pd.MultiIndex.from_frame(df[['ID', 'Month']]).isin(self.exclude_list)]

        return df
    
    def compare_metric_table(self,model):
        mat_metrics=self.get_metric_table("ground_truth")
        model_metrics=self.get_metric_table(model)

        return model_metrics.corrwith(mat_metrics)
        



# folder=r"C:\Users\franc\Box\Rehab Robotics Lab\Projects\PANDA Gym (# 834084)\Personnel\Students & RAs\Francis Sowande\Aim 3\Final Models\Model Data"

# pro=analyze_models(folder,"val",show_included=True)

# # model_object=pro.get_model_object("ST_GCN_param_est")
# pro.compare_metric_table("BI_LSTM_2L")

