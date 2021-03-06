
import streamlit as st
import pandas as pd


# Importacion de archivo
def uploadFile(id:str):
    typeFilesSupport = ['csv', 'xls', 'xlsx', 'json']  # Tipos soportados

    uploadFile = st.file_uploader(label=id, type=typeFilesSupport)
    df = None

    if uploadFile is not None:
        try:

            if uploadFile.type == "text/csv":
                df = pd.read_csv(uploadFile)

            if uploadFile.type == "application/vnd.ms-excel":
                df = pd.read_excel(uploadFile)

            if uploadFile.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                df = pd.read_excel(uploadFile)
            
            if uploadFile.type == 'application/json':
                df = pd.read_json(uploadFile)

        except:
            st.error("El Archivo No Se puede Analizar")

    return df
