   
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 19:50:56 2022

@author: monicaagredalopez
"""
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


 #####VALIDATION DATASET########
def stats_plots_valid(phase,color,target,unit,case): 
    My_target   = target
    My_Phase    = phase  
    My_case     = case
    My_color    = color
    My_unit     = unit

    
    #Load R2 RMSE of the models- Valid dataset no bias correction   
    my_scores_valid = pd.read_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                                  key = My_Phase+'_'+My_case +'_'+ My_target+'_scores')


    #Plot of the histograms for R2 and RMSE of the models.
    fig, (ax,ax2) = plt.subplots(nrows=1,ncols=2,constrained_layout=True, figsize=(8, 4))
    
    ax.hist(my_scores_valid['r2_score'], bins='auto' ,density=True,  color= My_color, 
            alpha = 0.7,edgecolor = '#000000', label = 'Hist. distribution')
    ax.axvline(my_scores_valid['r2_score'].median(),
                color='black', label=r'Median = {:.2f}'.format(my_scores_valid['r2_score'].median()), 
                linestyle='--', linewidth=2)
    ax.set_xlabel(r'R$^2$')
    ax.set_ylabel('Probability Density')
    ax.set_title(My_case)
    ax.legend() 
    
  
    ax2.hist(my_scores_valid['root_mean_squared_error'], bins='auto', density=True, color= My_color ,
             alpha = 0.7,edgecolor = '#000000',label = 'Hist. distribution')
    ax2.axvline(my_scores_valid['root_mean_squared_error'].median(),
                color='black', label=r'Median = {:.0f} '.format(my_scores_valid['root_mean_squared_error'].median()) + My_unit, 
                linestyle='--', linewidth=2)
    ax2.set_xlabel('RMSE')
    ax2.set_ylabel('Probability Density')
    ax2.set_title(My_case)
    ax2.legend(loc='best')
    plt.suptitle(My_target)
    plt.show()




start=datetime.now()            

phases2 = ['Clinopyroxene']
colors  = ['#00ABB3'] 
targets = ['P_kbar','T_C']
units   = ['kbar','°C']
cases   = ['cpx_only','cpx_liq','cpx_only_pwlr','cpx_liq_pwlr']


for Phase, Color in zip(phases2, colors):
    for Target, Unit in zip(targets, units):
        scores_test_f = pd.DataFrame()
        col_name    = []
        bias_c      = []
        r2_mean     = []
        r2_median   = []
        RMSE_mean   = []
        RMSE_median = []

        for Case in cases :
            stats_plots_valid(phase=Phase,color=Color,target=Target,unit=Unit,case=Case)
         
print(datetime.now()-start)
