
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
def stats_plots(phase,color,ecolor,target,tolerance_prec,tolerance_acc,unit,case): 
    My_target         = target
    My_Phase          = phase  
    My_case           = case
    My_color          = color
    My_ecolor         = ecolor
    My_tolerance_prec = tolerance_prec
    My_tolerance_acc  = tolerance_acc
    My_unit            = unit
    


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
  
    
    #Errors of the final model build using n=500 models
    r2_medians   = r2_score(my_test_y,my_predictions['Median'])
    RMSE_medians = mean_squared_error(my_test_y,my_predictions['Median'], squared=False) 

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
    my_predictions['q001s_bc_RF']          = my_predictions['q001s'] + pred_onx_tot['RF_predicted_biases']
    my_predictions['q999s_bc_RF']          = my_predictions['q999s'] + pred_onx_tot['RF_predicted_biases']
    my_predictions['RF_predicted_biases']  = pred_onx_tot['RF_predicted_biases']
    my_predictions_stats_bc = my_predictions[['Mean_bc_RF','Median_bc_RF','q001s_bc_RF','q999s_bc_RF',My_target + '_Y_test']] 
    my_predictions_stats_bc.to_excel(My_target+'_Results_Stats_bias_c_TEST.xlsx')
    

    r2_medians_bc   = r2_score(my_test_y,my_predictions['Median_bc_RF'])
    RMSE_medians_bc = mean_squared_error(my_test_y,my_predictions['Median_bc_RF'], squared=False) 

    #Post-filter
    #Precission
    ##No bias corrected
    My_predict_no_precise=my_predictions[(my_predictions['Median']-my_predictions['q001s'] >= My_tolerance_prec)&\
                                (my_predictions['q999s'] -my_predictions['Median']>= My_tolerance_prec)] 
    My_predict_precise=my_predictions[~my_predictions.indexes_test.isin(My_predict_no_precise.indexes_test)] 
    
   
    ##bias corrected B(X)-RF
    My_predict_no_precise_bc=my_predictions[(my_predictions['Median_bc_RF']-my_predictions['q001s_bc_RF'] >= My_tolerance_prec)&\
                                (my_predictions['q999s_bc_RF'] -my_predictions['Median_bc_RF']>= My_tolerance_prec)] 
    My_predict_precise_bc=my_predictions[~my_predictions.index.isin(My_predict_no_precise_bc.index)] 
      
  

    #Accuracy using biases as cutoff parameter
    ##No bias corrected
    My_predict_no_accurate_bias=my_predictions[(abs(my_predictions['RF_predicted_biases'])) 
                                           >=  My_tolerance_acc]
    My_predict_accurate_bias=my_predictions[~my_predictions.indexes_test.isin(My_predict_no_accurate_bias.indexes_test)]  
    
    
    ##bias corrected B(X)-RF
    My_predict_no_accurate_bias_bc=my_predictions[(abs(my_predictions['RF_predicted_biases'])) 
                                           >=  My_tolerance_acc]
    My_predict_accurate_bias_bc=my_predictions[~my_predictions.indexes_test.isin(My_predict_no_accurate_bias_bc.indexes_test)]  
    
    
    #Precission and accuracy
    ##No bias corrected
    My_predict_no_prec_acc = my_predictions[my_predictions.indexes_test.isin(My_predict_no_precise.indexes_test)|
                                            my_predictions.indexes_test.isin(My_predict_no_accurate_bias.indexes_test)]

    My_predict_prec_acc    = my_predictions[~my_predictions.indexes_test.isin(My_predict_no_prec_acc.indexes_test)]  

    
   
    ##bias corrected B(X)-RF
    My_predict_no_prec_acc_bc = my_predictions[my_predictions.indexes_test.isin(My_predict_no_precise_bc.indexes_test)|
                                               my_predictions.indexes_test.isin(My_predict_no_accurate_bias_bc.indexes_test)]

    My_predict_prec_acc_bc    = my_predictions[~my_predictions.indexes_test.isin(My_predict_no_prec_acc_bc.indexes_test)]  

    
    r2_medians_prec_acc_bc   = r2_score(My_predict_prec_acc_bc[My_target + '_Y_test'],My_predict_prec_acc_bc['Median_bc_RF'])
    RMSE_medians_prec_acc_bc = mean_squared_error(My_predict_prec_acc_bc[My_target + '_Y_test'],My_predict_prec_acc_bc['Median_bc_RF'], squared=False) 


    
    #FINAL PLOT 
    ##No bias->biasc-c->filtered
    ###No bias
    fig, (ax,ax2,ax3) = plt.subplots(nrows=1,ncols=3, constrained_layout=True, figsize=(15, 5))
    fig.suptitle('Test dataset: '+My_case, fontsize=18,)  #y=0.95
    ax.errorbar(my_predictions[My_target + '_Y_test'], my_predictions['Median'], 
                 yerr=[round(my_predictions['Median']-my_predictions['q001s'],5), 
                       round(my_predictions['q999s']-my_predictions['Median'],5)],
                 xerr=None, linestyle='', marker = 'o',capsize=0,
                 color = My_color, markersize = 5, markeredgecolor='black',markeredgewidth=0.4,elinewidth=1,
                 ecolor = My_ecolor,label = r'Median with 99% C.I.')
   
    ax.plot((my_predictions[My_target + '_Y_test'].min(),my_predictions[My_target + '_Y_test'].max()),
            (my_predictions[My_target + '_Y_test'].min(),my_predictions[My_target + '_Y_test'].max()), linestyle = '--', color ='black')                   
    ax.set_xlabel('Expected '+My_target)
    ax.set_ylabel('Predicted '+My_target)
    ax.set_title('No Bias corrected')
    ax.legend(title = r'R$^2$ = {:.2f}'.format(r2_medians) + ' - ' +  r'RMSE = {:.0f} '.format(RMSE_medians) 
              +My_unit  + ' - '+'n='+str(len(my_predictions[My_target + '_Y_test'])) ) 
    ###Bias-corrected 
    ax2.errorbar(My_predict_prec_acc_bc[My_target + '_Y_test'], My_predict_prec_acc_bc['Median_bc_RF'], 
                 yerr=[round(My_predict_prec_acc_bc['Median_bc_RF']- My_predict_prec_acc_bc['q001s_bc_RF'],5), 
                       round( My_predict_prec_acc_bc['q999s_bc_RF']- My_predict_prec_acc_bc['Median_bc_RF'],5)],
                 xerr=None, linestyle='', marker = 'o',capsize=0,
                 color = My_color, markersize = 5, markeredgecolor='black',markeredgewidth=0.4,elinewidth=1, 
                 ecolor = My_ecolor,label =r'Prec. ${<}$ '+  str(My_tolerance_prec) +' ' + My_unit   +' - '+ r'Acc.${<}$ ' + str( My_tolerance_acc)+' ' + My_unit)
    ax2.errorbar(My_predict_no_prec_acc[My_target + '_Y_test'],  My_predict_no_prec_acc['Median_bc_RF'], 
                 yerr=[round( My_predict_no_prec_acc_bc['Median_bc_RF']- My_predict_no_prec_acc_bc['q001s_bc_RF'],5), 
                       round( My_predict_no_prec_acc_bc['q999s_bc_RF']- My_predict_no_prec_acc_bc['Median_bc_RF'],5)],
                 xerr=None, linestyle='', marker = 'o',capsize=2,
                 color = '#8B1874', markersize = 5, markeredgecolor='black',markeredgewidth=0.4,elinewidth=1, 
                 ecolor = '#E11299',label = r'Prec. ${\geq}$ '+  str(My_tolerance_prec) +' ' + My_unit   +' - '+ r'Acc.${\geq}$ ' + str( My_tolerance_acc)+' ' + My_unit)
    
    
    ax2.plot((my_predictions[My_target + '_Y_test'].min(),my_predictions[My_target + '_Y_test'].max()),
            (my_predictions[My_target + '_Y_test'].min(),my_predictions[My_target + '_Y_test'].max()), linestyle = '--', color ='black')                   
    ax2.set_xlabel('Expected '+My_target)
    ax2.set_ylabel('Predicted '+My_target)
    ax2.set_title('B(X) RF-prec-acc-filter')
    ax2.legend(title = r'R$^2$ = {:.2f}'.format(r2_medians_bc) + ' - ' +  
               r'RMSE = {:.0f} '.format(RMSE_medians_bc) +My_unit
               + ' - '+'n='+str(len(my_predictions[My_target + '_Y_test'])))
    
    ###Bias corrected and filtered
    ax3.errorbar(My_predict_prec_acc_bc[My_target + '_Y_test'], My_predict_prec_acc_bc['Median_bc_RF'], 
                 yerr=[round(My_predict_prec_acc_bc['Median_bc_RF']- My_predict_prec_acc_bc['q001s_bc_RF'],5), 
                       round(My_predict_prec_acc_bc['q999s_bc_RF']- My_predict_prec_acc_bc['Median_bc_RF'],5)],
                 xerr=None, linestyle='', marker = 'o',capsize=0,
                 color = My_color, markersize = 5, markeredgecolor='black',markeredgewidth=0.4,elinewidth=1,
                 ecolor = My_ecolor,label = r'Median with 99% C.I.')
    
    
    ax3.plot((my_predictions[My_target + '_Y_test'].min(),my_predictions[My_target + '_Y_test'].max()),
            (my_predictions[My_target + '_Y_test'].min(),my_predictions[My_target + '_Y_test'].max()), linestyle = '--', color ='black')                   
    ax3.set_xlabel('Expected '+My_target)
    ax3.set_ylabel('Predicted '+My_target)
    ax3.set_title('B(X) RF-prec-acc-filtered')
    ax3.legend(title = r'R$^2$ = {:.2f}'.format(r2_medians_prec_acc_bc )
               + ' - ' +  r'RMSE = {:.0f} '.format(RMSE_medians_prec_acc_bc) +My_unit
               + ' - '+'n='+str(len(My_predict_prec_acc_bc[My_target + '_Y_test'])))



start=datetime.now()            
phases2 = ['Clinopyroxene']
colors  = ['#00ABB3']
ecolors = ['#B2B2B2']
targets = ['P_kbar','T_C']
units = ['kbar','°C']
tolerances_prec = [10,50]#kbar,C
tolerances_acc = [10,50]#kbar,C
cases   = ['cpx_only','cpx_liq']
for Phase, Color,Ecolor,  in zip(phases2, colors,ecolors):
    for Target, Unit, Tolerance_prec, Tolerance_acc in zip(targets, units,tolerances_prec,tolerances_acc):      
        for Case in cases:
            stats_plots(phase=Phase,color=Color,ecolor=Ecolor,tolerance_prec=Tolerance_prec,tolerance_acc=Tolerance_acc,
                        unit=Unit,target=Target,case=Case)
            
print(datetime.now()-start)

