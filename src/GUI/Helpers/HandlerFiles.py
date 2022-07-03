import streamlit as st
import pandas as pd


#Importacion de archivo


def uploadFile():
    labelFiles = "Escoge tu Archivo a Analizar"             #Label para UI upload file
    typeFilesSupport = ['csv', 'xls', 'xlsx', 'json']       #Tipos soportados

    uploadFile = st.file_uploader(label=labelFiles, type=typeFilesSupport)

    #print(uploadFile.type) 


