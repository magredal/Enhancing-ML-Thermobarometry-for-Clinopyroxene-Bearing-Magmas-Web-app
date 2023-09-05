#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:18:17 2022

@author: monicaagredalopez
"""

#Apply the models to the TEST dataset
import os
import joblib
import onnxruntime as rt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error 


def apply_to_test(phase,target,case):
    My_target = target
    My_Phase = phase  
    My_case = case


    #Load the test dataset
    X_test             = pd.read_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                                     key= My_Phase + '_'+My_case +'_'+ My_target+'_X_test')
    indexes_test       = pd.read_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                                     key= My_Phase + '_'+My_case+'_'+ My_target + '_indexes_test')
    y_test             = pd.read_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                                     key= My_Phase + '_'+My_case +'_'+ My_target+ '_Y_test')
   
    #Upload the std scaler model and scale the test
    Std_scaler_model = joblib.load('std-scalers/std_scaler_'+My_case+'_'+ My_target +'.bin')
    X_test_s    = Std_scaler_model.transform(X_test) #"_s = scaled"
    
    #Upload and apply the models to the test data 
    currpath = os.getcwd()
    new_path = currpath +  '/' 'models'+'/'+My_Phase +'/'+My_case
    n=500 #Number of trained models (see from script 1.)
    
    #Load and run the models using ONNX Runtime
    Inx = indexes_test
    pred_onx_tot = Inx.copy()
  
    
    r2_test = []
    RMSE_test = []

    
    for i in range(n):

        models     = rt.InferenceSession(new_path+'/'+My_target+'_'+My_Phase+'_'+My_case+'_model_'+str(i)+'.onnx')
        input_name = models.get_inputs()[0].name  
        label_name = models.get_outputs()[0].name 
        pred_onx   = models.run([label_name],{input_name: X_test_s.astype(np.float32)})[0]
        
        pred_onx_tot = pred_onx_tot.copy()
        pred_onx_tot['model_'+str(i)] = pred_onx.astype(np.float32) 
       
        
    
        #Evaluate the results of each model using the R2 and RMSE        
        r2_test.append(r2_score(y_test.loc[:,My_target], pred_onx[:,0])) #P_kbar
        RMSE_test.append(mean_squared_error(y_test.loc[:,My_target],pred_onx[:,0],squared=False)) #squared=False returns RMSE insteasd of MSE
        
        
    #Save the predictions
    pred_onx_tot.to_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                     key= My_target+'_'+My_Phase+'_'+My_case + '_Y_pred' , format = 'table')    
       
   
    #Save r2 RMSE of each model 
    my_scores_test = {'r2_score': r2_test, 
                 'root_mean_squared_error': RMSE_test}
    my_scores_test_pd = pd.DataFrame.from_dict(my_scores_test) 
    my_scores_test_pd.to_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                               key =  My_target+'_'+My_Phase+'_'+My_case + '_y_test_scores', format = 'table')
    
    
    #Save the scaled X_test
    X_test_s_pd = pd.DataFrame.from_dict(X_test_s) #"_s = scaled"
    X_test_s_pd.to_hdf('out-files/'+My_Phase+'_'+My_case+'_'+ My_target+'.h5',
                     key= My_target+'_'+My_Phase+'_'+My_case + '_X_test_s_pred' , format = 'table')   
    
   
start=datetime.now()            
phases2 = ['Clinopyroxene']
targets = ['SiO2_Liq', 'TiO2_Liq', 'Al2O3_Liq', 'FeOt_Liq', 'MgO_Liq', 'MnO_Liq', 'CaO_Liq', 'Na2O_Liq', 'K2O_Liq']
cases   = ['cpx_only']
            
for Phase in phases2:
    print(Phase)
    for Target in targets:
        print(Target)
        for Case in cases:
            print(Case)  
            apply_to_test(phase=Phase,target=Target,case=Case)

print(datetime.now()-start)  

