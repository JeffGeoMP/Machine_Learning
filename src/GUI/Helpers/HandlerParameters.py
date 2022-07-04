from pandas import DataFrame
import streamlit as st



#Operaciones para las regresiones Lineales y Polinomiales
def OperationsRegressions(id:str):
    options = (
        "1. Grafica de Puntos",
        "2. Funcion de Tendencia Lineal/Polinomial",
        "3. Prediccion de Tendencia"
    )
    optionSelect = st.selectbox(id, options, 0)
    return options.index(optionSelect)


def SelectAxis(id:str, df:DataFrame):
    options = df.columns.values
    optionSelect = st.selectbox(id, options, 0)
    return optionSelect


def NumberInput():
    number = st.number_input("Ingrese valor a predecir")
    return number