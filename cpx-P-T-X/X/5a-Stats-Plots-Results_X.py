#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 19:50:56 2022

@author: monicaagredalopez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib 
import onnxruntime as rt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error 
from datetime import datetime

 #####TEST DATASET########

def stats_plots(phase,color,ecolor,target,target_name,tolerance_prec,tolerance_acc,unit,case): 
    My_target         = target
    My_target_name    = target_name
    My_Phase          = phase  
    My_case           = case
    My_color          = color
    My_ecolor         = ecolor
    My_tolerance_prec = tolerance_prec
    My_tolerance_acc  = tolerance_acc
    My_unit           = unit
    


    #Loading predicted values
    my_predictions = pd.read_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                                 key= My_target+'_'+My_Phase+'_'+My_case + '_Y_pred' )
    my_test_y      = pd.read_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                                 key= My_Phase + '_'+My_case +'_'+ My_target+ '_Y_test')
    

    
    #Calculating stats for the TEST 
    my_predictions_means   = my_predictions.loc[:, my_predictions.columns!='indexes_test'].mean(axis=1) #axis = 1 means rows
    my_predictions_stds    = my_predictions.loc[:, my_predictions.columns!='indexes_test'].std(axis=1) #axis=1 means std of the rows
    my_predictions_medians = my_predictions.loc[:, my_predictions.columns!='indexes_test'].median(axis=1) 
    q3, q1                 = np.percentile(my_predictions.loc[:, my_predictions.columns!='indexes_test'], [75 ,25], axis=1)
    q001s, q999s           = np.percentile(my_predictions.loc[:, my_predictions.columns!='indexes_test'], [1 ,99], axis=1)
    my_predictions_iqrs    = q3 - q1
    
    
    #Adding rows to the df
    my_predictions['Mean']                = my_predictions_means
    my_predictions['Median']              = my_predictions_medians
    my_predictions['Std']                 = my_predictions_stds
    my_predictions['q001s']               = q001s
    my_predictions['q999s']               = q999s  
    my_predictions['IQR']                 = my_predictions_iqrs
    my_predictions[My_target + '_Y_test'] = my_test_y
    my_predictions_stats = my_predictions[['Mean','Median','Std','q001s','q999s',My_target + '_Y_test']] 
    my_predictions_stats.to_excel(My_target+'Stats_Test_no_bias_c.xlsx')
    
    
    # Bias corrections to test dataset
    # B(X) Random Forest
    #Load the Unknown (test) dataset
    X_test               = pd.read_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                         key= My_Phase + '_'+My_case +'_'+ My_target+'_X_test')
    my_test_indexes_pd   = pd.read_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                                       key= My_Phase + '_'+My_case+'_'+ My_target + '_indexes_test')   
    my_col_names_ordered = pd.read_hdf('models-bias-correction/'+My_Phase +'_'+My_case+'_plus_bias.h5',
                                       key= My_target+'_'+My_Phase +'_'+My_case + '_my_columns_B_X')
    X_test               = X_test.loc[:, my_col_names_ordered.my_columns.tolist()]
    
    Std_scaler_model     = joblib.load('std-scalers/'+My_target+'_'+My_Phase+'_'+My_case+ '_std_scaler_plus_bias_B(X).bin') 
    X_test_scaled        = Std_scaler_model.transform(X_test) 
    
    
    
    Inx = my_test_indexes_pd
    pred_onx_tot = Inx.copy()
    models = rt.InferenceSession('models-bias-correction/model_' + My_target+'_'+My_Phase+'_'+My_case+'_bias_correction_B(X).onnx')
    input_name = models.get_inputs()[0].name  
    label_name = models.get_outputs()[0].name 
    pred_onx = models.run([label_name],
                        {input_name: X_test_scaled.astype(np.float32)})[0] 
    
    pred_onx_tot['RF_predicted_biases'] = pred_onx.astype(np.float32) 
    
    
    
    my_predictions['Mean_bc_RF']           = my_predictions['Mean'] + pred_onx_tot['RF_predicted_biases']
    my_predictions['Median_bc_RF']         = my_predictions['Median'] + pred_onx_tot['RF_predicted_biases']
    my_predictions['Std_bc_RF']            = my_predictions['Std'] + pred_onx_tot['RF_predicted_biases']#check this out
    my_predictions['q001s_bc_RF']          = my_predictions['q001s'] + pred_onx_tot['RF_predicted_biases']
    my_predictions['q999s_bc_RF']          = my_predictions['q999s'] + pred_onx_tot['RF_predicted_biases']
    my_predictions['RF_predicted_biases']  = pred_onx_tot['RF_predicted_biases']
    
    
   
    #Errors filter
    #Precission
    ##No bias corrected
    My_predict_no_precise=my_predictions[(my_predictions['Median']-my_predictions['q001s'] >= My_tolerance_prec)&\
                                (my_predictions['q999s'] -my_predictions['Median']>= My_tolerance_prec)] 
    My_predict_precise=my_predictions[~my_predictions.indexes_test.isin(My_predict_no_precise.indexes_test)] 
    
    r2_medians_precs_f   = r2_score(My_predict_precise[My_target + '_Y_test'],My_predict_precise['Median'])
    RMSE_medians_precs_f = mean_squared_error(My_predict_precise[My_target + '_Y_test'],My_predict_precise['Median'], squared=False) 
    
    ## bias corrected B(X)-RF
    My_predict_no_precise_bc=my_predictions[(my_predictions['Median_bc_RF']-my_predictions['q001s_bc_RF'] >= My_tolerance_prec)&\
                                (my_predictions['q999s_bc_RF'] -my_predictions['Median_bc_RF']>= My_tolerance_prec)] 
    My_predict_precise_bc=my_predictions[~my_predictions.index.isin(My_predict_no_precise_bc.index)] 
      
    r2_medians_bc_precs_f   = r2_score(My_predict_precise_bc[My_target + '_Y_test'],My_predict_precise_bc['Median_bc_RF'])
    RMSE_medians_bc_precs_f = mean_squared_error(My_predict_precise_bc[My_target + '_Y_test'],My_predict_precise_bc['Median_bc_RF'], squared=False) 

    
 

    #Accuracy using biases as cutoff parameter
    ##No bias corrected
    My_predict_no_accurate_bias=my_predictions[(abs(my_predictions['RF_predicted_biases'])) 
                                           >=  My_tolerance_acc]
    My_predict_accurate_bias=my_predictions[~my_predictions.indexes_test.isin(My_predict_no_accurate_bias.indexes_test)]  
    
    r2_medians_acc_f   = r2_score(My_predict_accurate_bias[My_target + '_Y_test'],My_predict_accurate_bias['Median'])
    RMSE_medians_acc_f = mean_squared_error(My_predict_accurate_bias[My_target + '_Y_test'],My_predict_accurate_bias['Median'], squared=False) 

    ##bias corrected B(X)-RF
    My_predict_no_accurate_bias_bc=my_predictions[(abs(my_predictions['RF_predicted_biases'])) 
                                           >=  My_tolerance_acc]
    My_predict_accurate_bias_bc=my_predictions[~my_predictions.indexes_test.isin(My_predict_no_accurate_bias_bc.indexes_test)]  
    
    r2_medians_bc_acc_f   = r2_score(My_predict_accurate_bias_bc[My_target + '_Y_test'],My_predict_accurate_bias_bc['Median_bc_RF'])
    RMSE_medians_bc_acc_f = mean_squared_error(My_predict_accurate_bias_bc[My_target + '_Y_test'],My_predict_accurate_bias_bc['Median_bc_RF'], squared=False) 


    #Precission and accuracy
    ##No bias corrected
    My_predict_no_prec_acc = my_predictions[my_predictions.indexes_test.isin(My_predict_no_precise.indexes_test)|
                                            my_predictions.indexes_test.isin(My_predict_no_accurate_bias.indexes_test)]

    My_predict_prec_acc    = my_predictions[~my_predictions.indexes_test.isin(My_predict_no_prec_acc.indexes_test)]  
   
    
    r2_medians_prec_acc_f   = r2_score(My_predict_prec_acc[My_target + '_Y_test'],My_predict_prec_acc['Median'])
    RMSE_medians_prec_acc_f = mean_squared_error(My_predict_prec_acc[My_target + '_Y_test'],My_predict_prec_acc['Median'], squared=False) 

    ##bias corrected B(X)-RF
    My_predict_no_prec_acc_bc = my_predictions[my_predictions.indexes_test.isin(My_predict_no_precise_bc.indexes_test)|
                                               my_predictions.indexes_test.isin(My_predict_no_accurate_bias_bc.indexes_test)]
    
    My_predict_prec_acc_bc    = my_predictions[~my_predictions.indexes_test.isin(My_predict_no_prec_acc_bc.indexes_test)]  
  
    
    r2_medians_prec_acc_bc   = r2_score(My_predict_prec_acc_bc[My_target + '_Y_test'],My_predict_prec_acc_bc['Median_bc_RF'])
    RMSE_medians_prec_acc_bc = mean_squared_error(My_predict_prec_acc_bc[My_target + '_Y_test'],My_predict_prec_acc_bc['Median_bc_RF'], squared=False) 

    

    #Final figures
    ax0.errorbar(My_predict_prec_acc_bc[My_target + '_Y_test'], My_predict_prec_acc_bc['Median_bc_RF'], 
                 yerr=[round(My_predict_prec_acc_bc['Median_bc_RF']- My_predict_prec_acc_bc['q001s_bc_RF'],5), 
                       round(My_predict_prec_acc_bc['q999s_bc_RF']- My_predict_prec_acc_bc['Median_bc_RF'],5)],
                 xerr=None, linestyle='', marker = 'o',capsize=0,
                 color = My_color, markersize = 5, markeredgecolor='black',markeredgewidth=0.4,elinewidth=1,
                 ecolor = My_ecolor,label = r'Median with 99% C.I.')
    
    
    
    ax0.plot((My_predict_prec_acc_bc[My_target + '_Y_test'].min(),My_predict_prec_acc_bc[My_target + '_Y_test'].max()),
            (My_predict_prec_acc_bc[My_target + '_Y_test'].min(),My_predict_prec_acc_bc[My_target + '_Y_test'].max()), linestyle = '--', color ='black')  
                 
    ax0.set_xlabel('Expected '+My_target_name+' [' +My_unit +']', fontsize='11')
    ax0.set_ylabel('Predicted '+My_target_name +' [' +My_unit +']', fontsize='11')
    ax0.set_title('B(X) RF-prec-acc-filtered')
    ax0.legend(title = r'R$^2$ = {:.2f}'.format(r2_medians_prec_acc_bc )
               + ' - ' +  r'RMSE = {:.2f} '.format(RMSE_medians_prec_acc_bc) +My_unit
               + ' - '+'n='+str(len(My_predict_prec_acc_bc[My_target + '_Y_test'])))
   
    plt.tight_layout()

   


start=datetime.now()            

phases2 = ['Clinopyroxene']
colors  = ['#00ABB3']
ecolors = ['#B2B2B2']
targets = ['SiO2_Liq', 'TiO2_Liq', 'Al2O3_Liq', 'FeOt_Liq', 'MgO_Liq', 'MnO_Liq', 'CaO_Liq', 'Na2O_Liq', 'K2O_Liq']
targets_names = [r'SiO$_2$ Liq', r'TiO$_2$ Liq', r'Al$_2$O$_3$ Liq', r'FeOt Liq', r'MgO Liq', r'MnO Liq', r'CaO Liq', r'Na$_2$O Liq', r'K$_2$O Liq']
units = ['wt %','wt %','wt %','wt %','wt %','wt %','wt %','wt %','wt %']
tolerances_prec = [1.5,0.3,0.5,0.8,0.6,0.02,0.6,0.25,0.3]
tolerances_acc = [5,0.5,1,3,3,0.05,1,1,1]
I= [0,1,2,3,4,5,6,7,8]
cases   = ['cpx_only']

plt.figure(figsize= [15,15])
for Phase, Color,Ecolor,  in zip(phases2, colors,ecolors):
    plt.subplots_adjust(top=0.934, bottom=0.046, left=0.045, right=0.990,hspace=0.4, wspace=0.4)
    plt.suptitle('Test dataset: Liquid', fontsize= 14)
   
    for i, Target, Target_names, Unit, Tolerance_prec, Tolerance_acc in zip(I,targets,targets_names, units,tolerances_prec,tolerances_acc):      
        ax0 = plt.subplot(3,3,i+1)
        for Case in cases:
            stats_plots(phase=Phase,color=Color,ecolor=Ecolor,tolerance_prec=Tolerance_prec,tolerance_acc=Tolerance_acc,
                        unit=Unit,target=Target, target_name=Target_names,case=Case)
            
print(datetime.now()-start)

