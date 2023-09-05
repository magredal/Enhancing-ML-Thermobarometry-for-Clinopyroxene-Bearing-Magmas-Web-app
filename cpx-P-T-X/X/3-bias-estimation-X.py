import pandas as pd
import os
from datetime import datetime

def bias_estimation(phase,target,case): 
    My_target = target
    My_Phase = phase  
    My_case = case

    #Determine the bias on each sample (validation dataset)
    #Load validation dataset indexes

    #Load Global training-validation dataset indexes
    my_data = pd.read_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                          My_Phase + '_'+My_case +'_'+ My_target +  '_indexes_train_valid')

    
    my_indexes = my_data['indexes_train_valid'].values
    my_biases = []
    my_new_indexes = []

    for my_index in my_indexes:
        n = 500 #models (As in script 1)
        k = 0 #number of times you find the index in the validation dataset
        dev = 0
        for i in range (n): 
            #Load validation dataset indexes
            my_indexes_valid = pd.read_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                               key= My_Phase+'_'+My_case +'_'+ My_target+ '_indexes_valid_' + str(i))
            #Load validation dataset y
            my_ys_valid      = pd.read_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                               key= My_Phase+'_'+My_case +'_'+ My_target+'_y_valid_' + str(i))
            #Load validation dataset predictions
            my_ress_valid    = pd.read_hdf('out-files/'+My_Phase+'_'+My_case+'.h5',
                               key= My_Phase+'_'+My_case +'_'+ My_target+ '_res_' + str(i))
            
            #print(my_indexes_valid)
            my_indexes_valid = my_indexes_valid['indexes_valid'].tolist()
            my_ys_valid      = my_ys_valid[My_target].tolist()
            my_ress_valid    = my_ress_valid['prediction'].tolist()
            
           
            for my_index_valid, my_y_valid, my_res_valid in zip(my_indexes_valid, my_ys_valid, my_ress_valid):
               
                if my_index_valid == my_index:
                    dev = dev + my_y_valid - my_res_valid #bias
                    k = k+1
        if k > 0:
                
            average_bias=(dev/k)
            print(average_bias)
        else:
            average_bias=0
            print('NOT FOUND!!!')
            
        my_biases.append(average_bias)  
        my_new_indexes.append(my_index)     
    
    my_data['new_indexes'] = my_new_indexes
    my_data['biases']      = my_biases
    print(len(my_data.index))
    my_data                = my_data[my_data['biases'] !=0]
    print(len(my_data.index))
    
   #To save
    currpath = os.getcwd()
    new_path = currpath +'/' 'models-bias-correction'
    if not os.path.exists(new_path):
        os.mkdir(new_path)
        
    my_data.to_hdf(new_path+ '/'+ My_Phase+'_'+My_case+'_plus_bias.h5',
                     key = My_target+'_'+My_Phase +'_'+My_case +'_plus_bias' , format = 'table')   


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
            bias_estimation(phase=Phase,target=Target,case=Case)
            
print(datetime.now()-start)