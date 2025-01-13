import streamlit as st
import pandas as pd
from PIL import Image
import os

from functions.predictions import *
from functions.functions import *



  ## SETTING AND HEADER   ###

im = Image.open("imgs/logo.png")

st.set_page_config(
    page_title="ML cpx Thermobarometry",
    page_icon=im,
    layout="wide"
)


im2 = Image.open("imgs/logo.png")

col1, col2 = st.columns([1.2, 1])
with col1:
    st.title("ML cpx Thermobarometry")
    #st.header("Welcome to our Thermobarometry Web App!")
    st.write(
        "In this webapp, we introduce our easy-to-use tool for estimating Pressure ($P$) and Temperature ($T$) using our Machine-Learning (ML) based clinopyroxene (Cpx) and Clinopyroxene-liquid (Cpx-liq) thermobarometers. The thermobarometers used here follow the Machine-Learning workflow presented in Ágreda-López et al. (2024).")
    
with col2:
    st.image(im2, width=350)
    
st.header("How it Works")
st.write(        
"**Choose Your Model:** Select either Cpx or Cpx-liq based on your needs.\n\n\
**Input Your Data:** Simply download our template and fill in your own values. Leave unknown values blank.\n\n\
**Upload Your Data:** Once your template is filled out, upload it to our system.\n\n\
**Add Analytical Uncertainty:** If you have information on the analytical uncertainty of each oxide, please provide it in the corresponding box. If not, you can use the default values.\n\n\
**Get Your Results:** Download your results.\n\n\
**Note:** The user is responsible for pre-processing their own data (e.g., quality data filters, equilibrium tests).")

## INPUT##

st.header("Input")

cpx = st.radio(
    "What's your input?",
    ["cpx_only", "cpx_liquid"],
    horizontal=True
)    

if cpx == "cpx_only":
    
    Elements = ['SiO2_Cpx', 'TiO2_Cpx', 'Al2O3_Cpx', 'FeOt_Cpx', 'MgO_Cpx', 'MnO_Cpx', 'CaO_Cpx',  'Na2O_Cpx', 'Cr2O3_Cpx']
    Elements_std = ['SiO2_Cpx_rel_err', 'TiO2_Cpx_rel_err', 'Al2O3_Cpx_rel_err', 'FeOt_Cpx_rel_err', 'MgO_Cpx_rel_err', 'MnO_Cpx_rel_err', 'CaO_Cpx_rel_err',
                    'Na2O_Cpx_rel_err', 'Cr2O3_Cpx_rel_err']

    st.markdown("The input of the model must have the following structure (it is not necessary to keep the same order of the columns):")           
    input_example =  pd.read_excel('files/template_cpx.xlsx')
    st.dataframe(input_example)
                
    st.markdown("Select the button below if you want to download an empty file with the correct structure.")
                
                
    df_input_sheet = pd.read_excel('files/template_cpx_empty.xlsx')
    df_input_sheet_xlsx = to_excel(df_input_sheet)
    st.download_button(label='Download the input file form', data=df_input_sheet_xlsx , file_name= 'template_cpx_empty.xlsx')
    
    
    ## ERROR MANAGEMENT ##

    st.header("Analytical Uncertainties")

    st.markdown("Set relative errors for input data (E.g., 0.01 means 1%)")

    # Errors typically associated to each oxide measurements in the EPMA
    std_dev_perc_default = [0.03,0.08,0.03,0.03,0.03,0.08,0.03,0.08,0.08]
    std_dev_perc = [0.03,0.08,0.03,0.03,0.03,0.08,0.03,0.08,0.08]

    if 'SiO2_Cpx' not in st.session_state:
        st.session_state["SiO2_Cpx"] = std_dev_perc_default[0]
    if 'TiO2_Cpx' not in st.session_state:
        st.session_state["TiO2_Cpx"] = std_dev_perc_default[1]
    if 'Al2O3_Cpx' not in st.session_state:
        st.session_state["Al2O3_Cpx"] = std_dev_perc_default[2]
    if 'FeOt_Cpx' not in st.session_state:
        st.session_state["FeOt_Cpx"] = std_dev_perc_default[3]
    if 'MgO_Cpx' not in st.session_state:
        st.session_state["MgO_Cpx"] = std_dev_perc_default[4]
    if 'MnO_Cpx' not in st.session_state:
        st.session_state["MnO_Cpx"] = std_dev_perc_default[5]
    if 'CaO_Cpx' not in st.session_state:
        st.session_state["CaO_Cpx"] = std_dev_perc_default[6]
    if 'Na2O_Cpx' not in st.session_state:
        st.session_state["Na2O_Cpx"] = std_dev_perc_default[7]
    if 'Cr2O3_Cpx' not in st.session_state:
        st.session_state["Cr2O3_Cpx"] = std_dev_perc_default[8]
    
    st.session_state["SiO2_Cpx_Error"] = st.session_state["SiO2_Cpx"]
    st.session_state["TiO2_Cpx_Error"] = st.session_state["TiO2_Cpx"]
    st.session_state["Al2O3_Cpx_Error"] = st.session_state["Al2O3_Cpx"]
    st.session_state["FeOt_Cpx_Error"] = st.session_state["FeOt_Cpx"]
    st.session_state["MgO_Cpx_Error"] = st.session_state["MgO_Cpx"]
    st.session_state["MnO_Cpx_Error"] = st.session_state["MnO_Cpx"]
    st.session_state["CaO_Cpx_Error"] = st.session_state["CaO_Cpx"]
    st.session_state["Na2O_Cpx_Error"] = st.session_state["Na2O_Cpx"]
    st.session_state["Cr2O3_Cpx_Error"] = st.session_state["Cr2O3_Cpx"]


    std = st.radio(
    "Relative errors",
    ["Equal for all observations", "Different for each observation"],
    horizontal=True
    )

    if std == 'Equal for all observations':

        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
        with c1:
            std_dev_perc[0] = st.number_input("SiO2_Cpx_Error (0.03)", key="SiO2_Cpx_Error", on_change=save_value_cpx, step=1e-4, format="%.3f")
            std_dev_perc[1] = st.number_input('TiO2_Cpx_Error (0.08)', key="TiO2_Cpx_Error", on_change=save_value_cpx, step=1e-4, format="%.3f")
        with c2:
            std_dev_perc[2] = st.number_input('Al2O3_Cpx_Error (0.03)', key="Al2O3_Cpx_Error", on_change=save_value_cpx, step=1e-4, format="%.3f")
            std_dev_perc[3] = st.number_input('FeOt_Cpx_Error (0.03)', key="FeOt_Cpx_Error", on_change=save_value_cpx, step=1e-4, format="%.3f")
        with c3:
            std_dev_perc[4] = st.number_input('MgO_Cpx_Error (0.03)', key="MgO_Cpx_Error", on_change=save_value_cpx, step=1e-4, format="%.3f")
            std_dev_perc[5] = st.number_input('MnO_Cpx_Error (0.08)', key="MnO_Cpx_Error", on_change=save_value_cpx, step=1e-4, format="%.3f")
        with c4:
            std_dev_perc[6] = st.number_input('CaO_Cpx_Error (0.03)', key="CaO_Cpx_Error", on_change=save_value_cpx, step=1e-4, format="%.3f")
            std_dev_perc[7] = st.number_input('Na2O_Cpx_Error (0.08)', key="Na2O_Cpx_Error", on_change=save_value_cpx, step=1e-4, format="%.3f")
        with c5:
            std_dev_perc[8] = st.number_input('Cr2O3_Cpx_Error (0.08)', key="Cr2O3_Cpx_Error", on_change=save_value_cpx, step=1e-4, format="%.3f")

    elif std == 'Different for each observation':

        st.markdown("Select the button below if you want to download an empty file with the correct structure for relative errors.")
                
                
        df_std_sheet = pd.read_excel('files/template_cpx_rel_err_empty.xlsx')
        df_std_sheet_xlsx = to_excel(df_std_sheet)
        st.download_button(label='Download the errors file form', data=df_std_sheet_xlsx , file_name= 'template_cpx_rel_err_empty.xlsx')

        st.markdown("Upload a file:")
    
        uploaded_std = st.file_uploader("Choose a file for relative errors")

        if uploaded_std is not None:
            filename = uploaded_std.name
            nametuple = os.path.splitext(filename)
        
            if nametuple[1] == '.csv':
                # read csv
                df_std = pd.read_csv(uploaded_std)
                st.dataframe(df_std)
            elif nametuple[1] == '.xls' or nametuple[1] == '.xlsx':
                # read xls or xlsx
                df_std = pd.read_excel(uploaded_std)
                st.dataframe(df_std)
            else:
                st.warning("Incorrect file type (you need to upload a csv, xls or xlsx file)")

    ## PROCESSING##
      
    st.header("Processing")
    
    st.markdown("Upload a file:")
    
    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        filename = uploaded_file.name
        nametuple = os.path.splitext(filename)
    
        if nametuple[1] == '.csv':
            # read csv
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
        elif nametuple[1] == '.xls' or nametuple[1] == '.xlsx':
            # read xls or xlsx
            df = pd.read_excel(uploaded_file)
            st.dataframe(df)
        else:
            st.warning("Incorrect file type (you need to upload a csv, xls or xlsx file)")


else:

    Elements = ['SiO2_Cpx', 'TiO2_Cpx', 'Al2O3_Cpx', 'FeOt_Cpx', 'MgO_Cpx', 'MnO_Cpx', 'CaO_Cpx',  'Na2O_Cpx', 'Cr2O3_Cpx',
                    'SiO2_Liq', 'TiO2_Liq', 'Al2O3_Liq', 'FeOt_Liq', 'MgO_Liq', 'MnO_Liq', 'CaO_Liq',  'Na2O_Liq', 'K2O_Liq']
    Elements_std = ['SiO2_Cpx_std', 'TiO2_Cpx_std', 'Al2O3_Cpx_std', 'FeOt_Cpx_std', 'MgO_Cpx_std', 'MnO_Cpx_std', 'CaO_Cpx_std',
                    'Na2O_Cpx_std', 'Cr2O3_Cpx_std', 'SiO2_Liq_std', 'TiO2_Liq_std', 'Al2O3_Liq_std', 'FeOt_Liq_std', 'MgO_Liq_std',
                    'MnO_Liq_std', 'CaO_Liq_std',  'Na2O_Liq_std', 'K2O_Liq_std']


    st.markdown("The input of the model must have the following structure (it is not necessary to keep the same order of the columns):")           
    input_example =  pd.read_excel('files/template_cpx_liq.xlsx')
    st.dataframe(input_example)
                
    st.markdown("Select the button below if you want to download an empty file with the correct structure.")
                
                
    df_input_sheet = pd.read_excel('files/template_cpx_liq_empty.xlsx')
    df_input_sheet_xlsx = to_excel(df_input_sheet)
    st.download_button(label='Download the input file form', data=df_input_sheet_xlsx , file_name= 'template_cpx_liq_empty.xlsx')
    
    
    ## ERROR MANAGEMENT ##

    st.header("Analytical Uncertainties")

    st.markdown("Set relative errors for input data (E.g., 0.01 means 1%)")

    # Errors typically associated to each oxide measures in the EPMA
    std_dev_perc_default = [0.03,0.08,0.03,0.03,0.03,0.08,0.03,0.08,0.08,0.03,0.08,0.03,0.03,0.03,0.08,0.03,0.08,0.08]
    std_dev_perc = [0.03,0.08,0.03,0.03,0.03,0.08,0.03,0.08,0.08,0.03,0.08,0.03,0.03,0.03,0.08,0.03,0.08,0.08]

    if 'SiO2_Cpx' not in st.session_state:
        st.session_state["SiO2_Cpx"] = std_dev_perc_default[0]
    if 'TiO2_Cpx' not in st.session_state:
        st.session_state["TiO2_Cpx"] = std_dev_perc_default[1]
    if 'Al2O3_Cpx' not in st.session_state:
        st.session_state["Al2O3_Cpx"] = std_dev_perc_default[2]
    if 'FeOt_Cpx' not in st.session_state:
        st.session_state["FeOt_Cpx"] = std_dev_perc_default[3]
    if 'MgO_Cpx' not in st.session_state:
        st.session_state["MgO_Cpx"] = std_dev_perc_default[4]
    if 'MnO_Cpx' not in st.session_state:
        st.session_state["MnO_Cpx"] = std_dev_perc_default[5]
    if 'CaO_Cpx' not in st.session_state:
        st.session_state["CaO_Cpx"] = std_dev_perc_default[6]
    if 'Na2O_Cpx' not in st.session_state:
        st.session_state["Na2O_Cpx"] = std_dev_perc_default[7]
    if 'Cr2O3_Cpx' not in st.session_state:
        st.session_state["Cr2O3_Cpx"] = std_dev_perc_default[8]
    if 'SiO2_Liq' not in st.session_state:
        st.session_state["SiO2_Liq"] = std_dev_perc_default[9]
    if 'TiO2_Liq' not in st.session_state:
        st.session_state["TiO2_Liq"] = std_dev_perc_default[10]
    if 'Al2O3_Liq' not in st.session_state:
        st.session_state["Al2O3_Liq"] = std_dev_perc_default[11]
    if 'FeOt_Liq' not in st.session_state:
        st.session_state["FeOt_Liq"] = std_dev_perc_default[12]
    if 'MgO_Liq' not in st.session_state:
        st.session_state["MgO_Liq"] = std_dev_perc_default[13]
    if 'MnO_Liq' not in st.session_state:
        st.session_state["MnO_Liq"] = std_dev_perc_default[14]
    if 'CaO_Liq' not in st.session_state:
        st.session_state["CaO_Liq"] = std_dev_perc_default[15]
    if 'Na2O_Liq' not in st.session_state:
        st.session_state["Na2O_Liq"] = std_dev_perc_default[16]
    if 'K2O_Liq' not in st.session_state:
        st.session_state["K2O_Liq"] = std_dev_perc_default[17]
    
    st.session_state["SiO2_Cpx_Error"] = st.session_state["SiO2_Cpx"]
    st.session_state["TiO2_Cpx_Error"] = st.session_state["TiO2_Cpx"]
    st.session_state["Al2O3_Cpx_Error"] = st.session_state["Al2O3_Cpx"]
    st.session_state["FeOt_Cpx_Error"] = st.session_state["FeOt_Cpx"]
    st.session_state["MgO_Cpx_Error"] = st.session_state["MgO_Cpx"]
    st.session_state["MnO_Cpx_Error"] = st.session_state["MnO_Cpx"]
    st.session_state["CaO_Cpx_Error"] = st.session_state["CaO_Cpx"]
    st.session_state["Na2O_Cpx_Error"] = st.session_state["Na2O_Cpx"]
    st.session_state["Cr2O3_Cpx_Error"] = st.session_state["Cr2O3_Cpx"]
    st.session_state["SiO2_Liq_Error"] = st.session_state["SiO2_Liq"]
    st.session_state["TiO2_Liq_Error"] = st.session_state["TiO2_Liq"]
    st.session_state["Al2O3_Liq_Error"] = st.session_state["Al2O3_Liq"]
    st.session_state["FeOt_Liq_Error"] = st.session_state["FeOt_Liq"]
    st.session_state["MgO_Liq_Error"] = st.session_state["MgO_Liq"]
    st.session_state["MnO_Liq_Error"] = st.session_state["MnO_Liq"]
    st.session_state["CaO_Liq_Error"] = st.session_state["CaO_Liq"]
    st.session_state["Na2O_Liq_Error"] = st.session_state["Na2O_Liq"]
    st.session_state["K2O_Liq_Error"] = st.session_state["K2O_Liq"]

    std = st.radio(
    "Relative errors",
    ["Equal for all observations", "Different for each observation"],
    horizontal=True
    )

    if std == 'Equal for all observations':

        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
        with c1:
            std_dev_perc[0] = st.number_input("SiO2_Cpx_Error (0.03)", key="SiO2_Cpx_Error", on_change=save_value_cpx, step=1e-4, format="%.3f")
            std_dev_perc[1] = st.number_input('TiO2_Cpx_Error (0.08)', key="TiO2_Cpx_Error", on_change=save_value_cpx, step=1e-4, format="%.3f")
            std_dev_perc[2] = st.number_input('Al2O3_Cpx_Error (0.03)', key="Al2O3_Cpx_Error", on_change=save_value_cpx, step=1e-4, format="%.3f")
            std_dev_perc[3] = st.number_input('FeOt_Cpx_Error (0.03)', key="FeOt_Cpx_Error", on_change=save_value_cpx, step=1e-4, format="%.3f")
        with c2:
            std_dev_perc[4] = st.number_input('MgO_Cpx_Error (0.03)', key="MgO_Cpx_Error", on_change=save_value_cpx, step=1e-4, format="%.3f")
            std_dev_perc[5] = st.number_input('MnO_Cpx_Error (0.03)', key="MnO_Cpx_Error", on_change=save_value_cpx, step=1e-4, format="%.3f")
            std_dev_perc[6] = st.number_input('CaO_Cpx_Error (0.03)', key="CaO_Cpx_Error", on_change=save_value_cpx, step=1e-4, format="%.3f")
            std_dev_perc[7] = st.number_input('Na2O_Cpx_Error (0.08)', key="Na2O_Cpx_Error", on_change=save_value_cpx, step=1e-4, format="%.3f")
        with c3:
            std_dev_perc[8] = st.number_input('Cr2O3_Cpx_Error (0.08)', key="Cr2O3_Cpx_Error", on_change=save_value_cpx, step=1e-4, format="%.3f")
            std_dev_perc[9] = st.number_input("SiO2_Liq_Error (0.03)", key="SiO2_Liq_Error", on_change=save_value_liq, step=1e-4, format="%.3f")
            std_dev_perc[10] = st.number_input('TiO2_Liq_Error (0.08)', key="TiO2_Liq_Error", on_change=save_value_liq, step=1e-4, format="%.3f")
            std_dev_perc[11] = st.number_input('Al2O3_Liq_Error (0.03)', key="Al2O3_Liq_Error", on_change=save_value_liq, step=1e-4, format="%.3f")    
        with c4:
            std_dev_perc[12] = st.number_input('FeOt_Liq_Error (0.03)', key="FeOt_Liq_Error", on_change=save_value_liq, step=1e-4, format="%.3f")
            std_dev_perc[13] = st.number_input('MgO_Liq_Error (0.03)', key="MgO_Liq_Error", on_change=save_value_liq, step=1e-4, format="%.3f")
            std_dev_perc[14] = st.number_input('MnO_Liq_Error (0.08)', key="MnO_Liq_Error", on_change=save_value_liq, step=1e-4, format="%.3f")
            std_dev_perc[15] = st.number_input('CaO_Liq_Error (0.03)', key="CaO_Liq_Error", on_change=save_value_liq, step=1e-4, format="%.3f")      
        with c5:
            std_dev_perc[16] = st.number_input('Na2O_Liq_Error (0.08)', key="Na2O_Liq_Error", on_change=save_value_liq, step=1e-4, format="%.3f")
            std_dev_perc[17] = st.number_input('K2O_Liq_Error (0.08)', key="K2O_Liq_Error", on_change=save_value_liq, step=1e-4, format="%.3f")


    elif std == 'Different for each observation':

        st.markdown("Select the button below if you want to download an empty file with the correct structure for relative errors.")
                
                
        df_std_sheet = pd.read_excel('files/template_cpx_liq_rel_err_empty.xlsx')
        df_std_sheet_xlsx = to_excel(df_std_sheet)
        st.download_button(label='Download the errors file form', data=df_std_sheet_xlsx , file_name= 'template_cpx_liq_rel_err_empty.xlsx')

        st.markdown("Upload a file:")
    
        uploaded_std = st.file_uploader("Choose a file for relative errors")

        if uploaded_std is not None:
            filename = uploaded_std.name
            nametuple = os.path.splitext(filename)
        
            if nametuple[1] == '.csv':
                # read csv
                df_std = pd.read_csv(uploaded_std)
                st.dataframe(df_std)
            elif nametuple[1] == '.xls' or nametuple[1] == '.xlsx':
                # read xls or xlsx
                df_std = pd.read_excel(uploaded_std)
                st.dataframe(df_std)
            else:
                st.warning("Incorrect file type (you need to upload a csv, xls or xlsx file)")

    ## PROCESSING##
      
    st.header("Processing")
    
    st.markdown("Upload a file")
    
    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        filename = uploaded_file.name
        nametuple = os.path.splitext(filename)
    
        if nametuple[1] == '.csv':
            # read csv
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
        elif nametuple[1] == '.xls' or nametuple[1] == '.xlsx':
            # read xls or xlsx
            df = pd.read_excel(uploaded_file)
            st.dataframe(df)
        else:
            st.warning("Incorrect file type (you need to upload a csv, xls or xlsx file)")

if st.button('Make predictions'):
    
    if std == 'Equal for all observations':

        df_std = pd.DataFrame(np.repeat([std_dev_perc], repeats=len(df), axis=0), columns = Elements_std)

    else:

        if len(df_std)!=len(df) or len(df_std.columns)!=len(df.columns):

            st.warning("Input dataset and standard deviation has different size.")

    st.markdown(
    f'<p style="font-size:20px;border-radius:2%;">{"Predictions in progress..."}</p>',
    unsafe_allow_html=True) 
    
    # predict and show results
    try:
        df = df
    except:
        df = input_example


    df_output = predict(df,df_std,cpx)   
    st.write('Predicted values:')
    st.dataframe(df_output)
    
    
      
    df_xlsx_pred = to_excel(df_output, index=True)
    st.download_button(label='Download the output file',
                       data=df_xlsx_pred,
                       file_name='Predictions.xlsx')

    col1, col2 = st.columns(2)
    with col1:
        plothist(df_output)
    
