import os
import pandas as pd
import numpy as np
import Thermobar as pt

###0-Data pre-processing###
#Oxides for each phase
Elements  = {
  'Liquid':        ['SiO2', 'TiO2', 'Al2O3', 'FeOtot', 'MgO', 'MnO', 'CaO', 'Na2O', 'K2O','Cr2O3','NiO','P2O5'],
  'Clinopyroxene': ['SiO2', 'TiO2', 'Al2O3', 'FeOtot', 'MgO', 'MnO', 'CaO', 'Na2O', 'Cr2O3', 'K2O', 'NiO','P2O5']
  }

###Pre-processing###
def calculate_cations_on_oxygen_basis(myData0, myPhase, myElements, n_oxygens):
                        #Mol.we  , #cat #ani
    Weights = {'SiO2':  [60.0843,  1.0, 2.0],
               'TiO2':  [79.8788,  1.0, 2.0],
               'Al2O3': [101.961,  2.0, 3.0], 
               'FeOtot':[71.8464,  1.0, 1.0],
               'MgO':   [40.3044,  1.0, 1.0], 
               'MnO':   [70.9375,  1.0, 1.0],
               'CaO':   [56.0774,  1.0, 1.0],
               'Na2O':  [61.9789,  2.0, 1.0],
               'K2O':   [94.196,   2.0, 1.0],
               'Cr2O3': [151.9982, 2.0, 3.0],
               'P2O5':  [141.937,  2.0, 5.0], 
               'H2O':   [18.01388, 2.0, 1.0],
               'NiO':   [74.6928,  1.0, 1.0]}
    
    myData = myData0.copy()
    
    # Cation mole proportions
    for el in myElements:
        myData[el + '_cat_mol_prop'] = myData[el + '_' + myPhase] * Weights[el][1] / Weights[el][0]
        
    # Oxygen mole proportions
    for el in myElements:
        myData[el + '_oxy_mol_prop'] = myData[el + '_' + myPhase] * Weights[el][2] / Weights[el][0]
    
    # Oxigen mole proportions totals
    totals = np.zeros(len(myData.index))
    for el in myElements:
        totals += myData[el + '_oxy_mol_prop']
    
    myData['tot_oxy_prop'] = totals
    
    # tot_cations
    totals = np.zeros(len(myData.index))
    for el in myElements:
        myData[el + '_num_cat'] = n_oxygens * myData[el + '_cat_mol_prop']  /  myData['tot_oxy_prop']
        totals += myData[el + '_num_cat']
    
    return totals

def filter_by_cryst_formula(dataFrame, myPhase, myElements):
    
    Cations_Oxigens_Tolerance = {
                       #cat, oxigens, +- threshold on the cations
      'Olivine':       [3,    4,  0.0175],
      'Orthopyroxene': [4,    6,  0.025],
      'Clinopyroxene': [4,    6,  0.04],
      'Garnet':        [8,    12, 0.05],
      'Plagioclase':   [5,    8,  0.035],
      'Amphibole':     [15.5, 23, 0.5] 
    }
    
    dataFrame['Tot_cations'] = calculate_cations_on_oxygen_basis(myData0 = dataFrame, myPhase = myPhase, 
                                                                 myElements = myElements, 
                                                                 n_oxygens = Cations_Oxigens_Tolerance[myPhase][1])
    dataFrame = dataFrame[
        (dataFrame['Tot_cations'] < Cations_Oxigens_Tolerance[myPhase][0] + Cations_Oxigens_Tolerance[myPhase][2]) &                                               
        (dataFrame['Tot_cations'] > Cations_Oxigens_Tolerance[myPhase][0] - Cations_Oxigens_Tolerance[myPhase][2])]  
                                                
    dataFrame = dataFrame.drop(columns=['Tot_cations']) #removes the column 'Tot_cations from dataFrame
    return dataFrame
                                                

def adjustFeOtot(dataFrame):
    for i in range(len(dataFrame.index)):
        try:
            if pd.to_numeric(dataFrame.Fe2O3[i]) > 0:
                dataFrame.loc[i,'FeOtot'] = pd.to_numeric(dataFrame.FeO[i]) + 0.8998 * pd.to_numeric(dataFrame.Fe2O3[i])     
            else:
                dataFrame.loc[i,'FeOtot'] = pd.to_numeric(dataFrame.FeO[i]) 
        except:
            dataFrame.loc[i,'FeOtot'] = 0
            print('exception FeOtot = 0')
    return dataFrame                                                  
                                                   
    
def select_base_features(dataFrame, my_elements):
    dataFrame = dataFrame[my_elements]
    return dataFrame  
   
def data_imputation(dataFrame):
    dataFrame = dataFrame.fillna(0) 
    return dataFrame

def pwlr(dataFrame, my_phases):
    
    for my_phase in my_phases:
        my_indexes = []
        column_list = Elements[my_phase]
        
        for col in column_list:
            col = col + '_' + my_phase
            my_indexes.append(dataFrame.columns.get_loc(col))
            my_min = dataFrame[col][dataFrame[col] > 0].min() 
            dataFrame.loc[dataFrame[col] == 0,
                          col] = dataFrame[col].apply(
                              lambda x: np.random.uniform(
                                  np.nextafter(0.0, 1.0),my_min))
                              
        for ix in range(len(column_list)):
            for jx in range(ix+1, len(column_list)):
                col_name = 'log_' + dataFrame.columns[
                    my_indexes[jx]] + '_' + dataFrame.columns[
                        my_indexes[ix]] 
                dataFrame.loc[:,col_name] = np.log(
                    dataFrame[dataFrame.columns[my_indexes[jx]]]/ \
                    dataFrame[dataFrame.columns[my_indexes[ix]]]) 
    return dataFrame

phases2 = ['Clinopyroxene']
for My_Phase in phases2:
    print(My_Phase)
    def data_pre_processing(phase_1, phase_2, out_file): 
        try:
            os.remove(out_file) 
            
        except OSError:
            pass
        
        starting = pd.read_excel('cpx_dat_MAL_pt.xlsx', sheet_name='Experiment')
        starting.name = ''
        starting = starting[['Index', 'T_C', 'T_K' ,'P_kbar']]
        starting.to_hdf(out_file, key ='starting_material', format = 'table')
        
        phases = [phase_1, phase_2]
        for ix, my_phase in enumerate(phases):
            my_dataset = pd.read_excel('cpx_dat_MAL_pt.xlsx',sheet_name = my_phase)
            my_dataset = (my_dataset.
                          pipe(adjustFeOtot).
                          pipe(select_base_features ,
                               my_elements= Elements[my_phase]).
                          pipe(data_imputation))
            
            my_dataset = my_dataset.add_suffix('_' + my_phase)
            my_dataset.to_hdf(out_file, key=my_phase, format = 'table')
            
        my_phase_1 = pd.read_hdf(out_file, phase_1) 
        my_phase_2 = pd.read_hdf(out_file, phase_2)
        
        my_dataset = pd.concat([starting, my_phase_1 , my_phase_2], axis=1)
        
        my_dataset = my_dataset[(my_dataset['SiO2_Liquid'] > 35)&
                                (my_dataset['SiO2_Liquid'] < 80)] 

        my_dataset = my_dataset[(
            my_dataset['SiO2_'+phase_2] > 0)] 
    
        my_dataset = my_dataset[(my_dataset['P_kbar'] <= 30)]
        
        
        my_dataset = my_dataset[(my_dataset['T_C'] >= 650) & 
                                (my_dataset['T_C'] <= 1800)]
        
        
        my_dataset = filter_by_cryst_formula(dataFrame = my_dataset, 
                                             myPhase = phase_2 ,
                                             myElements = Elements[phase_2])
        
        my_dataset = my_dataset.sample(frac=1, random_state=50).reset_index(drop=True)
        
        my_labels = my_dataset[['Index', 'T_C', 'T_K' ,'P_kbar']]
        my_labels.to_hdf(out_file,  key='labels', format = 'table')
        
        my_dataset = my_dataset.drop(columns=['T_C', 'T_K' ,'P_kbar'])
        
        #phase-liq
        my_dataset.to_hdf(out_file, key= phase_1 + '_' + phase_2, format = 'table')

        
    currpath = os.getcwd()
    new_path = currpath +'/' 'out-files'
    if not os.path.exists(new_path):
        os.mkdir(new_path)           
    data_pre_processing(phase_1    ='Liquid' , 
                        phase_2    = My_Phase,
                        out_file   = new_path+'/'+My_Phase +'_Liquid_filtered.h5')#This is the main file that has all the info for each phase

###End pre-procesing###


#Equilibrium tests
#preparing the file for thermobar syntaxis

labels = pd.read_hdf('out-files/'+My_Phase +'_liquid_filtered.h5', key = 'labels')
cpx_liq = pd.read_hdf('out-files/'+My_Phase +'_liquid_filtered.h5', key = 'Liquid_' + My_Phase)
cpx_liq = cpx_liq.join(labels.set_index('Index'), on='Index')

def adjust_column_names_tb(dataFrame): 
    dataFrame.columns = [c.replace('_Liquid', '_Liq')
                         for c in dataFrame.columns] 
    dataFrame.columns = [c.replace('_Clinopyroxene', '_Cpx')
                        for c in dataFrame.columns]
    dataFrame.columns = [c.replace('FeOtot', 'FeOt')
                         for c in dataFrame.columns]
   
    return dataFrame

dataFrame = cpx_liq
cpx_liq_tb = adjust_column_names_tb(dataFrame)
cpx_liq_tb.to_excel('out-files/Clinopyroxene_Liquid_filtered_pt.xlsx')


out=pt.import_excel('out-files/Clinopyroxene_Liquid_filtered_pt.xlsx', sheet_name='Sheet1')
my_input =out['my_input']
Liqs =out['Liqs']
Cpxs =out['Cpxs']
PTs =out['Experimental_press_temp']
Ps = PTs['P_kbar']
Ts = PTs['T_K']
IDs = my_input['Index']


Eq_tests = pt.calculate_cpx_liq_eq_tests(liq_comps=Liqs, cpx_comps=Cpxs, Fe3Fet_Liq=None, P=Ps, T=Ts, sigma=1, Kd_Err=0.08) 
Eq_tests['Index'] = IDs
Eq_tests['P_kbar'] = Ps
Eq_tests['T_C'] = Ts-273.15
std_kd_FeOt = Eq_tests['Kd_Fe_Mg_Fet'].std()


Eq_tests_rm_Put2008 = Eq_tests[(Eq_tests['Delta_Kd_Put2008'] >= -0.08) & 
                         (Eq_tests['Delta_Kd_Put2008'] <= 0.08)]


#exporting and saving data ready to apply model
#Oxides that are gonna be used to calibrate the model
Elements_Cpx = ['SiO2_Cpx', 'TiO2_Cpx', 'Al2O3_Cpx', 'FeOt_Cpx', 'MgO_Cpx', 'MnO_Cpx', 'CaO_Cpx',  'Na2O_Cpx', 'Cr2O3_Cpx']
Elements_Liq = ['SiO2_Liq', 'TiO2_Liq', 'Al2O3_Liq', 'FeOt_Liq', 'MgO_Liq', 'MnO_Liq', 'CaO_Liq',  'Na2O_Liq', 'K2O_Liq']

#Saving cpx-only columns
Elements_Cpx_pd = pd.DataFrame(data=np.zeros([len(Elements_Cpx),len(Elements_Cpx)]),columns=Elements_Cpx)
Elements_Cpx_pd.to_hdf('out-files/'+My_Phase +'_Liquid_filtered.h5',
                       key='columns_names_cpx_only',format = 'table')
#Saving cpx-liq columns
Elements_Cpx_liq_pd = pd.DataFrame(data=np.zeros([len(Elements_Cpx+Elements_Liq),len(Elements_Cpx+Elements_Liq)]),
                                   columns=Elements_Cpx+Elements_Liq)
Elements_Cpx_liq_pd.to_hdf('out-files/'+My_Phase +'_Liquid_filtered.h5',
                       key='columns_names_cpx_liq',format = 'table' )


cols_to_keep = ['Index','P_kbar','T_C'] + Elements_Cpx+Elements_Liq

Eq_tests_rm_Put2008.loc[:,cols_to_keep].to_hdf('out-files/'+My_Phase +'_Liquid_filtered.h5', key = My_Phase + '_Liq_filtered_eqTest', format = 'table') 
Cpx_Liq_f = Eq_tests_rm_Put2008.loc[:,cols_to_keep]

#To calculate the pwlr
def pwlr2(dataFrame, column_list):
    
    #for my_phase in my_phases:
    my_indexes = []
    #To assign random values -no zeros- between zero and the min value for each oxide
    for col in column_list:
        my_indexes.append(dataFrame.columns.get_loc(col))
        my_min = dataFrame[col][dataFrame[col] > 0].min() 
        dataFrame.loc[dataFrame[col] == 0,
                      col] = dataFrame[col].apply(
                          lambda x: np.random.uniform(
                              np.nextafter(0.0, 1.0),my_min))
    #Calculate lrpwt 
    columns_name_pwlr = []                     
    for ix in range(len(column_list)):
        for jx in range(ix+1, len(column_list)):
            col_name = 'log_' + dataFrame.columns[
                my_indexes[jx]] + '_' + dataFrame.columns[
                    my_indexes[ix]] 
            columns_name_pwlr.append(col_name)
            dataFrame.loc[:,col_name] = np.log(
                dataFrame[dataFrame.columns[my_indexes[jx]]]/ \
                dataFrame[dataFrame.columns[my_indexes[ix]]]) 
    return dataFrame, columns_name_pwlr


#phase-liq-lrpwt 
Cpx_Liq_f_pwlr,columns_name_pwlr_liq = pwlr2(Cpx_Liq_f,Elements_Liq)
Cpx_Liq_f_pwlr,columns_name_pwlr_cpx = pwlr2(Cpx_Liq_f_pwlr,Elements_Cpx)


#Saving cpx-only-pwlr columns
columns_name_pwlr_cpx_pd = pd.DataFrame(data=np.zeros([len(Elements_Cpx+columns_name_pwlr_cpx),len(Elements_Cpx+columns_name_pwlr_cpx)]),
                                     columns=Elements_Cpx+columns_name_pwlr_cpx)
columns_name_pwlr_cpx_pd.to_hdf('out-files/'+My_Phase +'_Liquid_filtered.h5',
                       key='columns_names_cpx_only_pwlr',format = 'table')
#Saving cpx-liq-pwlr columns
columns_name_pwlr_cpx_liq_pd = pd.DataFrame(data=np.zeros([len(Elements_Cpx+Elements_Liq+columns_name_pwlr_cpx+columns_name_pwlr_liq),
                                                  len(Elements_Cpx+Elements_Liq+columns_name_pwlr_cpx+columns_name_pwlr_liq)]),
                                   columns=Elements_Cpx+Elements_Liq+columns_name_pwlr_cpx+columns_name_pwlr_liq)
columns_name_pwlr_cpx_liq_pd.to_hdf('out-files/'+My_Phase +'_Liquid_filtered.h5',
                       key='columns_names_cpx_liq_pwlr',format = 'table')


#To save the dataset
Cpx_Liq_f_pwlr.to_hdf('out-files/'+My_Phase +'_Liquid_filtered.h5', 
                      key = My_Phase + '_Liq_pwlr_filtered', format = 'table') 

####Balancing the data####
balancing = pd.read_hdf('out-files/'+My_Phase +'_Liquid_filtered.h5', 
                      key = My_Phase + '_Liq_pwlr_filtered')

low_p=balancing.loc[(balancing['P_kbar']<2.5)]


cutoff = 225
step= 1
appended_data = []
for i in np.arange(0,30.1,step):
    mySubDataset = balancing[(balancing['P_kbar']>=i)&(balancing['P_kbar']< i + step)]
    ln = len(mySubDataset['P_kbar'])
    if ln > cutoff:
        mySubDataset = mySubDataset.sample(cutoff)  

    appended_data.append(mySubDataset)
FinalDataset = pd.concat(appended_data)   


#To save the Final, filtered and balanced dataset
FinalDataset.to_hdf('out-files/'+My_Phase +'_Liquid_filtered.h5', key = My_Phase + '_Liq_pwlr_balanced', format = 'table') 





            
            
            
            
            
            

    

      
