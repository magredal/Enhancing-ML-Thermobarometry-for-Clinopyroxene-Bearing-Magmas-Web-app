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
    std_dev_rep = np.repeat([std_dev_perc], repeats=len(X_rep), axis=0)*X_rep
    np.random.seed(10)
    X_perturb = np.random.normal(X_rep, std_dev_rep)
    groups = np.repeat(np.arange(len(X)), repeats=n_perturbations, axis=0)
    return X_perturb, groups

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


#@st.cache
#def convert_df(df):
#    return df.to_csv().encode('utf-8')
#
#
#def get_base64(bin_file):
#    with open(bin_file, 'rb') as f:
#        data = f.read()
#    return base64.b64encode(data).decode()
#
#
#def set_png_as_page_bg(png_file, opacity=1):
#    bin_str = get_base64(png_file)
#    page_bg_img = '''
#    <style>
#    .stApp {
#    background-image: url("data:image/png;base64,%s");
#    background-size: cover;
#    background-opacity:opacity;
#    background-repeat: no-repeat;
#    background-attachment: scroll; # doesn't work
#    }
#    </style>
#    ''' % bin_str
#    st.markdown(page_bg_img, unsafe_allow_html=True)
#    return

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