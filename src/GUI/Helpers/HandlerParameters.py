from click import option
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

def SolverNeuralNetwork(id:str):
    options = (
        "lbfgs",
        "sgd",
        "adam"
    )
    optionSelect = st.selectbox(id, options, 0)
    return optionSelect


def SelectAxis(id:str, df:DataFrame):
    options = df.columns.values
    optionSelect = st.selectbox(id, options, 0)
    return optionSelect


def NumberInput():
    number = st.number_input("Ingrese valor a predecir", min_value=1)
    return number

def NumberInputInteger(id:str):
    number = st.number_input(id, min_value=1, step=1)
    return number


def SelectColumns(id:str, df:DataFrame):
    options = df.columns.values
    optionSelects = st.multiselect(id, options, options[0])
    return optionSelects
    
def SelectColumn(id:str, df:DataFrame):
    options = df.columns.values
    optionSelect = st.selectbox(id, options, 0)
    return optionSelect


    
    