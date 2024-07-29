import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import joblib
import onnxruntime as rt
import json


def data_imputation(dataFrame):
    dataFrame = dataFrame.fillna(0) 
    return dataFrame 


def replace_zeros(dataFrame, column_name):
    min_non_zero = dataFrame[dataFrame[column_name] > 0][column_name].min()
    epsilon = np.nextafter(0.0, 1.0)
    dataFrame.loc[dataFrame[column_name] == 0, column_name] = dataFrame[column_name].apply(
        lambda x: np.random.uniform(epsilon, min_non_zero) if x == 0 else x
    )
    return dataFrame  


def input_perturbation(X, std_dev_perc, n_perturbations=15):
    X_rep = np.repeat(X, repeats=n_perturbations, axis=0)
    std_dev_rep = np.repeat(std_dev_perc, repeats=n_perturbations, axis=0)*X_rep
    np.random.seed(10)
    X_perturb = np.random.normal(X_rep, std_dev_rep)
    groups = np.repeat(np.arange(len(X)), repeats=n_perturbations, axis=0)
    return X_perturb, groups


def save_value_cpx():
    st.session_state["SiO2_Cpx"] = st.session_state["SiO2_Cpx_Error"]
    st.session_state["TiO2_Cpx"] = st.session_state["TiO2_Cpx_Error"]
    st.session_state["Al2O3_Cpx"] = st.session_state["Al2O3_Cpx_Error"]
    st.session_state["FeOt_Cpx"] = st.session_state["FeOt_Cpx_Error"]
    st.session_state["MgO_Cpx"] = st.session_state["MgO_Cpx_Error"]
    st.session_state["MnO_Cpx"] = st.session_state["MnO_Cpx_Error"]
    st.session_state["CaO_Cpx"] = st.session_state["CaO_Cpx_Error"]
    st.session_state["Na2O_Cpx"] = st.session_state["Na2O_Cpx_Error"]
    st.session_state["Cr2O3_Cpx"] = st.session_state["Cr2O3_Cpx_Error"]


def save_value_liq():
    st.session_state["SiO2_Liq"] = st.session_state["SiO2_Liq_Error"]
    st.session_state["TiO2_Liq"] = st.session_state["TiO2_Liq_Error"]
    st.session_state["Al2O3_Liq"] = st.session_state["Al2O3_Liq_Error"]
    st.session_state["FeOt_Liq"] = st.session_state["FeOt_Liq_Error"]
    st.session_state["MgO_Liq"] = st.session_state["MgO_Liq_Error"]
    st.session_state["MnO_Liq"] = st.session_state["MnO_Liq_Error"]
    st.session_state["CaO_Liq"] = st.session_state["CaO_Liq_Error"]
    st.session_state["Na2O_Liq"] = st.session_state["Na2O_Liq_Error"]
    st.session_state["K2O_Liq"] = st.session_state["K2O_Liq_Error"]


def P_T_predictors(output, model):

    if output == 'Pressure' and model == 'cpx_only':
        scaler=joblib.load("models/"+"Model_cpx_only_P_bias.joblib")
        predictor = rt.InferenceSession('models/'+'Model_cpx_only_P_bias'+'.onnx')
        with open("models/"+'Model_cpx_only_P_bias'+'.json') as json_file:
            bias_json = eval(json.load(json_file))
    
    elif output == 'Temperature' and model == 'cpx_only':
        scaler=joblib.load("models/"+"Model_cpx_only_T_bias.joblib")
        predictor = rt.InferenceSession("models/"+'Model_cpx_only_T_bias'+'.onnx')
        with open("models/"+'Model_cpx_only_T_bias'+'.json') as json_file:
            bias_json = eval(json.load(json_file))
    
    elif output == 'Pressure' and model == 'cpx_liquid':
        scaler=joblib.load("models/"+"Model_cpx_liquid_P_bias.joblib")
        predictor = rt.InferenceSession("models/"+'Model_cpx_liquid_P_bias'+'.onnx')
        with open("models/"+'Model_cpx_liquid_P_bias'+'.json') as json_file:
            bias_json = eval(json.load(json_file))
        
    elif output == 'Temperature' and model == 'cpx_liquid':
        scaler=joblib.load("models/"+"Model_cpx_liquid_T_bias.joblib")
        predictor = rt.InferenceSession("models/"+'Model_cpx_liquid_T_bias'+'.onnx')
        with open("models/"+'Model_cpx_liquid_T_bias'+'.json') as json_file:
            bias_json = eval(json.load(json_file))
            
    return(scaler, predictor, bias_json)


def max_perc(x):
    return(np.percentile(x,84))


def min_perc(x):
    return(np.percentile(x,16))


def train_domain(df, cpx, output):
    if cpx == 'cpx_only' and output == 'pressure':
        warning = df.apply(lambda x: '' if 0.250000<=x['Al2O3_Cpx']<=25.760000
                                            and 0.000137<=x['Na2O_Cpx']<=7.760000
                                            and 0.400000<=x['CaO_Cpx']<=24.820000
                                        else 'Input out of bound', axis=1)  
    elif cpx == 'cpx_liquid' and output == 'pressure':
        warning = df.apply(lambda x: '' if 0.250000<=x['Al2O3_Cpx']<=25.760000
                                            and 0.000137<=x['Na2O_Cpx']<=7.760000
                                            and 0.030000<=x['MgO_Liq']<=18.587513
                                        else 'Input out of bound', axis=1) 
    elif cpx == 'cpx_only' and output == 'temperature':
        warning = df.apply(lambda x: '' if 0.760000<=x['MgO_Cpx']<=31.300000
                                            and 0.400000<=x['CaO_Cpx']<=24.820000
                                            and 0.290000<=x['Al2O3_Cpx']<=19.010000
                                            and 1.700000<=x['FeOt_Cpx']<=34.200000
                                            and 0.000179<=x['MnO_Cpx']<=2.980000
                                        else 'Input out of bound', axis=1)
    elif cpx == 'cpx_liquid' and output == 'temperature':
        warning = df.apply(lambda x: '' if 0.030000<=x['MgO_Liq']<=17.723561
                                        else 'Input out of bound', axis=1)
    return warning


def to_excel(df, index=False, startrow = 0):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=index, startrow=startrow, sheet_name='Sheet1')           
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.close()
    processed_data = output.getvalue()
    return processed_data


def plothist(df_output):
    targets = ['P (kbar)', 'T (C)']
    x_lab = ['Pressure [kbar]', 'Temperature [°C]']
    y_lab = ['Frequency', 'Frequency']
    color = ['#9BB0C1', '#D37676']
    titles = ['Pressure', 'Temperature']
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for tg in [0, 1]:
        ax[tg].hist(df_output[targets[tg]], edgecolor='black', color=color[tg], alpha=0.7)
        ax[tg].set_title(titles[tg], fontsize=13)
        ax[tg].set_xlabel(x_lab[tg], fontsize=13)
        ax[tg].set_ylabel(y_lab[tg], fontsize=13)
        ax[tg].grid(color='#B2B2B2', linestyle='--', linewidth=0.5, alpha=0.4)
    st.pyplot(fig)
