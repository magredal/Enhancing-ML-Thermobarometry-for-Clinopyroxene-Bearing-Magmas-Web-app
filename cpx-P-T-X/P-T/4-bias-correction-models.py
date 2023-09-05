#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:51:27 2022

@author: monicaagredalopez
"""

import pandas as pd
import numpy as np
import joblib 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from datetime import datetime


def bias_correction(Phase,Target,Case):
    My_target = Target
    My_Phase = Phase  
    My_case = Case
    
   #Importing the global training validation dataset 
    my_train_valid_indexes = pd.read_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                     key = My_Phase + '_'+My_case +'_'+ My_target +  '_indexes_train_valid' ) 
    
    X_train_valid = pd.read_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                     key = My_Phase + '_'+My_case +'_'+ My_target + '_X_train_valid' ) 
    
    y_train_valid = pd.read_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                     key = My_Phase + '_'+My_case +'_'+ My_target + '_Y_train_valid') 
   
    
    
    #concatenate global training validation dataset
    global_train_valid = pd.concat([my_train_valid_indexes.reset_index(drop=True),
                                    X_train_valid.reset_index(drop=True),
                                    y_train_valid.reset_index(drop=True)], axis=1)#axis=1-> concat. horizontally
    
    
    #Import the bias values for the validation dataset
    valid_indexes = pd.read_hdf('models-bias-correction/'+My_Phase+'_'+My_case+'_plus_bias.h5',
                     key = My_target+'_'+My_Phase +'_'+My_case +'_plus_bias') 
   
    phase = pd.DataFrame(global_train_valid.loc[global_train_valid['indexes_train_valid'].isin(valid_indexes['indexes_train_valid'])])
    phase = pd.concat([phase.reset_index(drop=True),valid_indexes.reset_index(drop=True)],axis=1)
    
    phase.to_hdf('out-files/Validation_'+My_Phase+ '_'+My_case+'_plus_bias.h5',
                     key= My_target+'_'+My_Phase +'_'+My_case , format = 'table')
    
    my_y = phase['biases'].values
    
    #####################################################################  B(Y)
    # Bias correction involving linear or polinomial regression of the target variable B(Y)
    for ix in [1,2,3]:
        z = np.polyfit(phase[My_target], phase['biases'], ix ) # 1-linear 2-power 3-cubic correction
        p = np.poly1d(z)
        np.save('models-bias-correction/polinomial_fittion_bias_order_'+My_target+'_'+My_case+'_'+ str(ix)+'.npy', p)
    
    #####################################################################  B(X)
    # Bias correction involving RANDOM FOREST B(X)
    X_phase = phase.drop(columns=['indexes_train_valid','new_indexes','biases', My_target]) 
    
    # Save column names to get the correct order in point 5
    X_Phase_columns = X_phase.columns #names of the columns
    my_columns = {'my_columns': X_Phase_columns} 
    my_columns_pd = pd.DataFrame.from_dict(my_columns)
    my_columns_pd.to_hdf('models-bias-correction/'+My_Phase +'_'+My_case+'_plus_bias.h5',
                     key= My_target+'_'+My_Phase +'_'+My_case + '_my_columns_B_X' , format = 'table')
    
    
    #To scale all the dataset before the training
    scaler = StandardScaler().fit(X_phase) 
    X_phase_s= scaler.transform(X_phase) #"_s = scaled"
    
    #save the scaler in joblib format      
    joblib.dump(scaler, 'std-scalers/'+My_target+'_'+My_Phase+'_'+My_case+ '_std_scaler_plus_bias_B(X).bin', compress=True)
    
    
    #Defining the regressor and training the algorithm
    regressor = ExtraTreesRegressor(n_estimators=450,
                                     max_features=1.0, n_jobs=-1).fit(
                                         X_phase_s, my_y) 
                                                             
   
    ##To save the model in ONNX
    initial_type = [('float_input', FloatTensorType([None, X_phase.shape[1]]))] 
    model_onx = convert_sklearn(regressor, initial_types=initial_type,
                          target_opset=12) 
    with open('models-bias-correction/model_' + My_target+'_'+My_Phase+'_'+My_case+'_bias_correction_B(X).onnx', 'wb') as f:
        f.write(model_onx.SerializeToString())
 
    ##################################################################### B(X, Y)     
    # Bias correction involving RANDOM FOREST B(X, Y)
    X_phase = phase.drop(columns=['indexes_train_valid','new_indexes','biases']) #B(X,Y) 
    X_Phase_columns = X_phase.columns #names of the columns
    my_columns = {'my_columns': X_Phase_columns} 
    my_columns_pd = pd.DataFrame.from_dict(my_columns)
    my_columns_pd.to_hdf('models-bias-correction/'+My_Phase +'_'+My_case+'_plus_bias.h5',
                     key= My_target+'_'+My_Phase +'_'+My_case + '_my_columns_B_X_Y' , format = 'table')

    #To scale all the dataset before the training
    scaler = StandardScaler().fit(X_phase) 
    X_phase_s= scaler.transform(X_phase) #"_s = scaled"
    
    #save the scaler in joblib format       
    joblib.dump(scaler,  'std-scalers/'+My_target+'_'+My_Phase+'_'+My_case+ '_std_scaler_plus_bias_B(X_Y).bin', compress=True)
    
    
    #Defining the regressor and training the algorithm
    regressor = ExtraTreesRegressor(n_estimators=450,
                                     max_features=1.0, n_jobs=-1).fit(
                                         X_phase_s, my_y) #n_estimators=450 #max_features=1 means all the feautures = all the oxides
                                         
                           
      
    ##To save the model in ONNX
    initial_type = [('float_input', FloatTensorType([None, X_phase.shape[1]]))] 
    model_onx = convert_sklearn(regressor, initial_types=initial_type,
                          target_opset=12) 
    with open('models-bias-correction/model_' + My_target+'_'+My_Phase+'_'+My_case+'_bias_correction_B(X_Y).onnx', 'wb') as f:
        f.write(model_onx.SerializeToString())
    


start=datetime.now()            
phases2 = ['Clinopyroxene']
targets = ['P_kbar','T_C']
cases   = ['cpx_only','cpx_liq','cpx_only_pwlr','cpx_liq_pwlr']
for Phase in phases2:
    for Target in targets:
        print(Target)
        for Case in cases:
            print(Case)
            bias_correction(Phase=Phase,Target=Target,Case=Case)
            
print(datetime.now()-start)
