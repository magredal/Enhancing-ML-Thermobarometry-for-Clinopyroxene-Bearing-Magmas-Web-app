import streamlit as st
import numpy as np
import pandas as pd
import json
from functions.functions import *


# Bias correction functions

def bias_f_line(x,a):
    return a*(x-x0)

def bias_f_temp(x,
                ang_left,popt_left,
                ang_right,popt_right):
    global x0
    if x<ang_left:
        x0 = ang_left
        return bias_f_line(x,popt_left)
    elif x>ang_right:
        x0 = ang_right
        return bias_f_line(x,popt_right)
    return 0.0    

bias_f = np.vectorize(bias_f_temp)

def predict(data,std_dev_perc,cpx):
    # Numbers of perturbations
    p = 1000

    if cpx == 'cpx_only':
        Elements = ['SiO2_Cpx', 'TiO2_Cpx', 'Al2O3_Cpx', 'FeOt_Cpx', 'MgO_Cpx', 'MnO_Cpx', 'CaO_Cpx',  'Na2O_Cpx', 'Cr2O3_Cpx']
        Elements_std = ['SiO2_Cpx_std', 'TiO2_Cpx_std', 'Al2O3_Cpx_std', 'FeOt_Cpx_std', 'MgO_Cpx_std', 'MnO_Cpx_std', 'CaO_Cpx_std',
                        'Na2O_Cpx_std', 'Cr2O3_Cpx_std']
    else:
        Elements = ['SiO2_Cpx', 'TiO2_Cpx', 'Al2O3_Cpx', 'FeOt_Cpx', 'MgO_Cpx', 'MnO_Cpx', 'CaO_Cpx',  'Na2O_Cpx', 'Cr2O3_Cpx',
                    'SiO2_Liq', 'TiO2_Liq', 'Al2O3_Liq', 'FeOt_Liq', 'MgO_Liq', 'MnO_Liq', 'CaO_Liq',  'Na2O_Liq', 'K2O_Liq']
        Elements_std = ['SiO2_Cpx_std', 'TiO2_Cpx_std', 'Al2O3_Cpx_std', 'FeOt_Cpx_std', 'MgO_Cpx_std', 'MnO_Cpx_std', 'CaO_Cpx_std',
                        'Na2O_Cpx_std', 'Cr2O3_Cpx_std', 'SiO2_Liq_std', 'TiO2_Liq_std', 'Al2O3_Liq_std', 'FeOt_Liq_std', 'MgO_Liq_std',
                        'MnO_Liq_std', 'CaO_Liq_std',  'Na2O_Liq_std', 'K2O_Liq_std']
    
    for element in Elements:
        df_m = replace_zeros(data.copy(), element)  

    sample_names = data['Sample_ID'] 

    Xd = df_m[Elements]
    X = np.array(Xd)
    Xd_std = std_dev_perc[Elements_std]
    X_std = np.array(Xd_std)

    X_perturb, groups = input_perturbation(X,X_std ,n_perturbations=p)


    for tg in [0, 1]:
        if tg == 0:
            scaler, predictor, bias_json = P_T_predictors('Pressure', cpx)
            X_perturb_s = scaler.transform(X_perturb)
            
            bias_popt_left = np.array(bias_json['slope']['left'])
            bias_popt_right = np.array(bias_json['slope']['right'])
            ang_left = bias_json['angle']['left']
            ang_right = bias_json['angle']['right']
        else:
            scaler, predictor, bias_json = P_T_predictors('Temperature', cpx)
            X_perturb_s = scaler.transform(X_perturb)
            
            bias_popt_left = np.array(bias_json['slope']['left'])
            bias_popt_right = np.array(bias_json['slope']['right'])
            ang_left = bias_json['angle']['left']
            ang_right = bias_json['angle']['right']
       

        targets = ['P (kbar)', 'T (C)']
        target = targets[tg]
        names_targets = ['pressure', 'temperature']
        names_target = names_targets[tg]
        
        # Add a placeholder
        latest_iteration = st.empty()
        st.write('Predicting ' + names_target +' ...')


        input_name = predictor.get_inputs()[0].name  
        label_name = predictor.get_outputs()[0].name 
        y_pred = predictor.run([label_name],{input_name: X_perturb_s.astype(np.float32)})[0]


        unique_y_pred = np.apply_along_axis(np.median,1, np.split(y_pred[:,0],len(y_pred)/p))
        unique_y_perc_max = np.apply_along_axis(max_perc,1, np.split(y_pred[:,0],len(y_pred)/p))
        unique_y_perc_min = np.apply_along_axis(min_perc,1, np.split(y_pred[:,0],len(y_pred)/p))

        bias_temp = bias_f(unique_y_pred,
                           ang_left, bias_popt_left, 
                           ang_right, bias_popt_right)

        unique_y_pred_temp = unique_y_pred - bias_temp
        unique_y_perc_max_temp = unique_y_perc_max - bias_temp
        unique_y_perc_min_temp =  unique_y_perc_min - bias_temp

        # Bound of the training set
        len_pred = len(unique_y_pred)
        
        if tg == 0 and cpx == 'cpx_only':
            max_bound = np.ones(len_pred) * 33.790000 
            min_bound = np.ones(len_pred) * 0.000000 
        elif tg == 0 and cpx == 'cpx_liquid':
            max_bound = np.ones(len_pred) * 32.800000 
            min_bound = np.ones(len_pred) * 0.000000 
        elif tg == 1 and cpx == 'cpx_only':
            max_bound = np.ones(len_pred) * 1892.710000 
            min_bound = np.ones(len_pred) * 587.630000 
        elif tg == 1 and cpx == 'cpx_liquid':
            max_bound = np.ones(len_pred) * 1800.020000 
            min_bound = np.ones(len_pred) * 654.660000 
                
        unique_y_pred = np.minimum(max_bound, np.maximum(min_bound, unique_y_pred_temp))
        unique_y_perc_max = np.minimum(max_bound, np.maximum(min_bound, unique_y_perc_max_temp))
        unique_y_perc_min = np.minimum(max_bound, np.maximum(min_bound, unique_y_perc_min_temp))
        
        error = (unique_y_perc_max - unique_y_perc_min)/2


        if tg == 0:
            df_output = pd.DataFrame(
                columns=['Sample_ID'] + [targets[0], '16_Percentile - ' + targets[0], '84_Percentile - ' + targets[0], targets[1]
                                        , '16_Percentile - ' + targets[1], '84_Percentile - ' + targets[1]])

            df_output['Sample_ID'] = sample_names

        
        df_output[target] = unique_y_pred
        df_output['16_Percentile - ' + target] = unique_y_perc_min
        df_output['84_Percentile - ' + target] = unique_y_perc_max

        if tg == 0:
            df_output['P_warning'] = train_domain(data, cpx, 'pressure')

        else:
            df_output['T_warning'] = train_domain(data, cpx, 'temperature')

    return df_output
   

    
    



