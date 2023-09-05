#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:51:27 2022

@author: monicaagredalopez
"""


####1-Model training####
import os
import pandas as pd
import numpy as np
import joblib 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error 
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from datetime import datetime

 
def monte_carlo_simulation(X, y, indexes, n, columns, phase, target, case):
        My_target = target
        My_Phase = phase  
        My_case = case
        r2   = []
        RMSE = []
        for i in range(n):
            my_res = {}
            X_train_s, X_valid_s, y_train, y_valid, \
                indexes_train, indexes_valid = train_test_split(
                    X, y, indexes, test_size=0.2) #train + validation here is the 80% of the total dataset as we removed a 20% for the "Validation dataset"
            
     
            
            #Defining the regressor and training the algorithm
            regressor = ExtraTreesRegressor(n_estimators=450,
                                            max_features=1.0, n_jobs=-1).fit(
                                                X_train_s, y_train)
                                                
            print(i) 
            
                    
            #Apply the trained model to the validation dataset                                 
            my_prediction = regressor.predict(X_valid_s) #This is the model applied to the validation dataset

        
            #to save training dataset
            my_indexes_train = {'indexes_train': indexes_train} 
            my_indexes_train_pd = pd.DataFrame.from_dict(my_indexes_train)
            my_indexes_train_pd.to_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                             key= My_Phase+'_'+My_case +'_'+ My_target+ '_indexes_train_' + str(i), format = 'table')
       
            my_y_train_pd = pd.DataFrame(y_train, columns =[My_target])
            my_y_train_pd.to_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                             key= My_Phase+'_'+My_case +'_'+ My_target+ '_Y_train_'+ str(i) , format = 'table')
            
            my_X_train_s_pd = pd.DataFrame(X_train_s, columns = columns) 
            my_X_train_s_pd.to_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                             key=My_Phase+'_'+My_case +'_'+ My_target+ '_X_train_s_'+ str(i) , format = 'table')
            
            

            #to save validation dataset
            my_indexes_valid = {'indexes_valid': indexes_valid} 
            my_indexes_valid_pd = pd.DataFrame.from_dict(my_indexes_valid)
            my_indexes_valid_pd.to_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                             key= My_Phase+'_'+My_case +'_'+ My_target+ '_indexes_valid_' + str(i), format = 'table')
            

            my_y_valid_pd = pd.DataFrame(y_valid,columns =[My_target])
            my_y_valid_pd.to_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                             key= My_Phase+'_'+My_case +'_'+ My_target+'_y_valid_' + str(i), format = 'table')
            

            my_X_valid_s_pd = pd.DataFrame(X_valid_s, columns = columns) 
            my_X_valid_s_pd.to_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                             key= My_Phase+'_'+My_case +'_'+ My_target+ '_X_valid_s_' + str(i), format = 'table')
            
            
            #to save the predictions 
            my_res =   {'indexes_valid'  : indexes_valid,
                           'prediction': my_prediction} #To save validation datasets + predictions
            my_res_pd = pd.DataFrame.from_dict(my_res)
            my_res_pd.to_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                             key= My_Phase+'_'+My_case +'_'+ My_target+ '_res_' + str(i), format = 'table')
            
     
            #to save the models
            #Converts the model to ONNX (to storage the model in the pc)
            currpath = os.getcwd()
            new_path = currpath +'/' 'models'
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            new_path = new_path + '/' + My_Phase
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            new_path = new_path + '/' + My_case
            if not os.path.exists(new_path):
                os.mkdir(new_path)
                

        
            ##To save the model in ONNX 
            initial_type = [('float_input', FloatTensorType([None, X_train_s.shape[1]]))] 
          
            model_onx = convert_sklearn(regressor, initial_types=initial_type,
                                        target_opset=12) 

            with open(new_path+'/'+My_target+'_'+My_Phase+'_'+My_case+'_model_'+str(i)+'.onnx', 'wb') as f:
                f.write(model_onx.SerializeToString())
                  
            #Evaluate the results using the R2 and RMSE 
            r2.append(r2_score(y_valid, my_prediction)) 
            RMSE.append(np.sqrt(mean_squared_error(y_valid,my_prediction)))
  
        my_scores = {'r2_score'               : r2, 
                     'root_mean_squared_error': RMSE}
        my_scores_pd = pd.DataFrame.from_dict(my_scores) 
        
        
        
        my_scores_pd.to_hdf('out-files/'+My_Phase+'_'+My_case+'.h5', 
                            key = My_Phase+'_'+My_case +'_'+ My_target+'_scores', format = 'table')     
#End Monte carlo simulation function        
         

#This is all happening before and it is used in the montecarlo loop
def train_model(phase,target,case):
    My_Phase  = phase
    My_target = target
    My_case   = case
    print(My_target)
    print(My_case)
    
   
    columns_name = pd.read_hdf('out-files/Clinopyroxene_Liquid_filtered.h5',
                               key='columns_names_'+My_case).columns 

        
    phase = pd.read_hdf('out-files/Clinopyroxene_Liquid_filtered.h5', 
                        key = My_Phase + '_Liq_pwlr_balanced') 
    
 
    my_y = phase[My_target].values
    my_indexes = phase['Index'].values 

    X_phase = phase[columns_name]
    X_Phase_columns = X_phase.columns #names of the columns
    
    my_columns = {'my_columns': X_Phase_columns} 
    my_columns_pd = pd.DataFrame.from_dict(my_columns)
    
  
    
    my_columns_pd.to_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                     key= My_Phase +'_'+My_case+ '_my_columns' , format = 'table')
   
    

    #Here we are taking out a 10% "test dataset" before entering to the Montecarlo loop
    X_train_valid, X_test, y_train_valid, y_test, \
        indexes_train_valid, indexes_test = train_test_split(
            X_phase, my_y, my_indexes, test_size=0.1)
    
    #to save final "test dataset"
    ##1.The indexes of the test
    my_test_indexes = {'indexes_test': indexes_test} 
    my_test_indexes_pd = pd.DataFrame.from_dict(my_test_indexes)
    
    my_test_indexes_pd.to_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                     key= My_Phase + '_'+My_case+'_'+ My_target + '_indexes_test' , format = 'table')
  
    
    ##2.The Y (P or T) of the test     
    y_test_pd = pd.DataFrame(y_test, columns =[My_target])
    y_test_pd.to_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                     key= My_Phase + '_'+My_case +'_'+ My_target+ '_Y_test' , format = 'table')
  
  
    ##3.The X (oxides) of the test
    X_test.to_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                   key= My_Phase + '_'+My_case +'_'+ My_target+'_X_test' , format = 'table')
    
                    
    #to save final "train_valid dataset"
    ##1.The indexes of the train_valid dataset
    my_train_valid_indexes = {'indexes_train_valid': indexes_train_valid} 
    my_train_valid_indexes_pd = pd.DataFrame.from_dict(my_train_valid_indexes)
    my_train_valid_indexes_pd.to_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                     key= My_Phase + '_'+My_case +'_'+ My_target +  '_indexes_train_valid' , format = 'table')
    
    
    ##2.The Y (P-T) of the train_valid dataset
    y_train_valid_pd = pd.DataFrame(y_train_valid, columns = [My_target])
    y_train_valid_pd.to_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                     key= My_Phase + '_'+My_case +'_'+ My_target + '_Y_train_valid' , format = 'table')
   

    
    ##3.The X (oxides) of the train_valid dataset
    X_train_valid.to_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                    key= My_Phase + '_'+My_case +'_'+ My_target + '_X_train_valid' , format = 'table')
    
    
    #To scale all the dataset before the training
    scaler = StandardScaler().fit(X_train_valid) 
    X_train_valid_s= scaler.transform(X_train_valid) #"_s = scaled"
    
    #save the scaler in joblib format   
    currpath = os.getcwd()   
  
    currpath = os.getcwd()
    new_path = currpath +'/' 'std-scalers'
    if not os.path.exists(new_path):
        os.mkdir(new_path)
        
    joblib.dump(scaler, new_path+'/'+'std_scaler_'+My_case+'_'+ My_target +'.bin', compress=True)


   
    monte_carlo_simulation(X = X_train_valid_s, y = y_train_valid,
                           indexes = indexes_train_valid,
                           n = 500, columns = X_Phase_columns, phase = My_Phase, target=My_target, case=My_case) #n =500

 
start=datetime.now()         
phases2 = ['Clinopyroxene']
targets = ['P_kbar','T_C']
cases   = ['cpx_only','cpx_liq','cpx_only_pwlr','cpx_liq_pwlr']

for Phase in phases2: 
    for Target in targets:
        for Case in cases:
            train_model(phase=Phase,target=Target, case=Case)
    
    
    
print(datetime.now()-start)




