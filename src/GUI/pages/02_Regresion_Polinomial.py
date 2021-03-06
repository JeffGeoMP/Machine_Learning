
import streamlit as st
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import Helpers.HandlerFiles as Files
import Helpers.HandlerParameters as Parameters


def information():
    st.sidebar.write("## Regresion Polinomial 馃搼")

    st.sidebar.write("驴Qu茅 es?")
    st.sidebar.write(
        "La regresi贸n polinomial es un modelo de an谩lisis de regresi贸n en el que la relaci贸n entre la variable independiente X y la variable dependiente " +
         "Y se modela con un polinomio de n-茅simo grado en X. La regresi贸n polinomial se ajusta a una relaci贸n no " +
         "lineal entre el valor de X y la media condicional correspondiente de Y, denotada `E [Y|X]` " 
         "Aunque la regresi贸n polinomial ajusta un modelo no lineal a los datos, como problema de estimaci贸n estad铆stica es lineal, en el sentido de que la funci贸n de " + 
         "regresi贸n `E [Y|X]` es lineal en los par谩metros desconocidos que se estiman a partir de " +
         "los datos. Por esta raz贸n, la regresi贸n polinomial se considera un caso especial de regresi贸n lineal m煤ltiple.")
    
    st.sidebar.write("Las variables explicativas (independientes) que resultan de la expansi贸n polinomial de las variables de \"l铆nea base\" se conocen como t茅rminos de grado superior. " +
                    "Estas variables tambi茅n se utilizan en entornos de clasificaci贸n.鈥?")


def algoritmo():
    information()
    st.title("Algoritmo - Regresi贸n Polinomial")

    st.subheader("Carga de Archivo")
    df = Files.uploadFile("Seleccione el Archivo")

    if df is not None:

        st.write("Informaci贸n Cargada")
        st.dataframe(df)

        st.markdown("""---""")
        st.subheader("Operaciones")

        operation = Parameters.OperationsRegressions("Seleccione la Operaci贸n a Realizar")

        st.markdown("""---""")
        st.subheader("Par谩metros")

        if operation == 0:
            columnNameX = Parameters.SelectAxis("Seleccione Eje X para Plotear Puntos", df)
            columnNameY = Parameters.SelectAxis("Seleccione Eje Y para Plotear Puntos", df)

            resButton = st.button("Ejecutar Algoritmo")
            if resButton :
                if columnNameX != columnNameY:

                    try:

                        X = df[columnNameX].values.reshape((-1, 1))
                        Y = df[columnNameY]

                        st.markdown("""---""")
                        st.subheader("Resultados")
                        st.write("Gr谩fica de Dispersi贸n")

                        fig, ax = plt.subplots(1,1)  #Cantidad de Graficas Filas y Columnas
                        fig.suptitle('Gr谩fica de Dispersi贸n\n' + columnNameX +" vs " + columnNameY, fontsize="10")
                        fig.text(.5, -0.025, "Dispersi贸n de Puntos, Seg煤n las Columnas Seleccionadas", style = 'italic', fontsize= 8, ha='center', color = "red")
                        ax.grid()
                        ax.set_xlabel(columnNameX)
                        ax.set_ylabel(columnNameY)
                        ax.scatter(X,Y, color='coral')

                        st.pyplot(fig)

                    except:
                        st.error("No se ha podido Ejecutar La Operaci贸n, Seleccione Otras Columnas y Vuelva a Intentar")

                else:
                    st.warning("Las Columnas Deben De Ser Diferentes")

        if operation == 1:
            columnNameX = Parameters.SelectAxis("Seleccione Eje X para Definir Funci贸n de Tendencia", df)
            columnNameY = Parameters.SelectAxis("Seleccione Eje Y para Definir Funci贸n de Tendencia", df)
            polyDegree = Parameters.NumberInputInteger("Ingrese el grado del Polinomio a Ajustar")
            
            resButton = st.button("Ejecutar Algoritmo")
            if resButton :
                if columnNameX != columnNameY:
                    if polyDegree > 0 : 
                        try:
                            X = df[columnNameX].values.reshape((-1,1))                        
                            Y = df[columnNameY]

                            polynomial = PolynomialFeatures(degree=polyDegree)
                            X_POLYNOMIAL = polynomial.fit_transform(X)          #Transformamos x para una funcion polinomial

                            #Entremamos Modelo
                            model = LinearRegression().fit(X_POLYNOMIAL, Y)
                            Y_POLYNOMIAL = model.predict(X_POLYNOMIAL)

                            #Errores en Prediccion
                            errorMean = round(mean_squared_error(Y, Y_POLYNOMIAL, squared=True) , 4)
                            r2 =  round(r2_score(Y, Y_POLYNOMIAL) , 4)

                            st.markdown("""---""")
                            st.subheader("Resultados")

                            modelStr = "y = {0} ".format(model.intercept_)
                            degreeX = 0
                            for i in model.coef_ : 
                                if i != 0 : 
                                    if degreeX == 1:
                                        modelStr += "{0} {1}x".format("+" if i>0 else "", i)
                                    else:
                                        modelStr += "{0} {1}x^{2} ".format("+" if i>0 else "", i, degreeX)
                                degreeX = degreeX + 1 

                            st.write("Modelo Polinomial De Grado: " + str(polyDegree))
                            st.latex(modelStr)

                            st.write("Errores del Modelo")
                            errorMeanStr = "Error Medio = {0}".format(errorMean)
                            r2Str = "R^2 = {0}".format(r2)

                            st.latex(errorMeanStr)
                            st.latex(r2Str)

                            st.write("Gr谩fica de Tendencia")

                            fig, ax = plt.subplots()
                            fig.suptitle('Gr谩fica de Tendencia\n' + columnNameX +" vs " + columnNameY, fontsize="10")
                            fig.text(.5, -0.025, "Tendencia de Puntos, Seg煤n las Columnas Seleccionadas", style = 'italic', fontsize= 8, ha='center', color = "red")
                            ax.grid()
                            ax.set_xlabel(columnNameX)
                            ax.set_ylabel(columnNameY)
                            ax.plot(X, Y_POLYNOMIAL, linewidth=3, color = "orangered")
                            ax.scatter(X,Y, color="coral")
                            st.pyplot(fig)

                        except:
                            st.error("No se ha podido Ejecutar La Operacion, Seleccione Otras Columnas y Vuelva a Intentar")

                    else:
                        st.warning("El Grado del Polinomio Debe Ser Mayor a 0")    
                else:
                    st.warning("Las Columnas Deben De Ser Diferentes")

        if operation == 2:
            columnNameX = Parameters.SelectAxis("Seleccione Eje X para Realizar Predicci贸n del Modelo", df)
            columnNameY = Parameters.SelectAxis("Seleccione Eje Y para Realizar Predicci贸n del Modelo", df)
            polyDegree = Parameters.NumberInputInteger("Ingrese el Grado del Polinomio para Ajustar la Prediccion")
            numberPredict = Parameters.NumberInput()

            resButton = st.button("Ejecutar Algoritmo")
            if resButton :
                if columnNameX != columnNameY:
                    try:
                        X = df[columnNameX].values.reshape((-1,1))                        
                        Y = df[columnNameY]
                            
                        polynomial = PolynomialFeatures(degree=polyDegree)
                        X_POLYNOMIAL = polynomial.fit_transform(X)          #Transformamos x para una funcion polinomial
    
                        #Entremamos Modelo
                        model = LinearRegression().fit(X_POLYNOMIAL, Y)
                        Y_POLYNOMIAL = model.predict(X_POLYNOMIAL)
    
                        #Errores en Prediccion
                        errorMean = round(mean_squared_error(Y, Y_POLYNOMIAL, squared=True) , 4)
                        r2 =  round(r2_score(Y, Y_POLYNOMIAL) , 4)
    
                        X_NEW_MIN = 0.0
                        X_NEW_MAX = float(numberPredict)  #para el calculo de la prediccion
    
                        X_NEW = np.linspace(X_NEW_MIN,X_NEW_MAX,50)
                        X_NEW = X_NEW[:,np.newaxis]
    
                        X_NEW_TRANSF = polynomial.fit_transform(X_NEW)
                        Y_NEW = model.predict(X_NEW_TRANSF)
                        
                        st.markdown("""---""")
                        st.subheader("Resultados")
    
                        modelStr = "y = {0} ".format(model.intercept_)
                        degreeX = 0
                        for i in model.coef_ : 
                            if i != 0 : 
                                if degreeX == 1:
                                    modelStr += "{0} {1}x".format("+" if i>0 else "", i)
                                else:
                                    modelStr += "{0} {1}x^{2} ".format("+" if i>0 else "", i, degreeX)
                            degreeX = degreeX + 1 
                            
                        st.write("Modelo Polinomial De Grado: " + str(polyDegree))
                        st.latex(modelStr)
    
                        st.write("Errores del Modelo")
                        errorMeanStr = "Error Medio = {0}".format(errorMean)
                        r2Str = "R^2 = {0}".format(r2)
    
                        st.latex(errorMeanStr)
                        st.latex(r2Str)
    
                        st.write("Predicci贸n")
                        predictionStr = "Prediccion \\rightarrow {0}".format(Y_NEW[Y_NEW.size-1])
                        st.latex(predictionStr)
    
                    except:
                        st.error("No se ha podido Ejecutar La Operaci贸n, Ingrese Nuevos Valores y Vuelva a Intentar")
                
                else:
                    st.warning("Las Columnas Deben De Ser Diferentes")



algoritmo()
