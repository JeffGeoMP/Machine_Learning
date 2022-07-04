
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import Helpers.HandlerFiles as Files
import Helpers.HandlerParameters as Parameters

PATH_FINAL_REPORT = "assets/LinearRegression.png"

def information():
    st.sidebar.write("## Regresion Lineal 📑")

    st.sidebar.write(
        "Este algoritmo es un método estadístico que permite resumir y estudiar las relaciones entre dos variables continuas cuantitativas.")
    st.sidebar.write(
        "La Regresión Lineal es una técnica paramétrica que se utiliza para predecir variables continuas, dependientes, dado un conjunto de variables independientes. ")
    st.sidebar.write("Es de naturaleza paramétrica porque hace ciertas suposiciones basadas en el conjunto de datos. Si el conjunto de datos sigue esas suposiciones," +
                     "la regresión arroja resultados increíbles, de lo contrario, tiene dificultades para proporcionar una precisión convincente." +
                     "Matemáticamente, la regresión usa una función lineal para aproximar o predecir la variable dependiente dada como: `y = mx + b`")


def algoritmo():
    information()
    st.subheader("Algoritmo")

    df = Files.uploadFile()

    if df is not None:
        operation = Parameters.OperationsRegressions(
            "Operaciones para Regresion Lineal")

        st.subheader("Parametros")

        if operation == 0:
            columnNameX = Parameters.SelectAxis(
                "Seleccione Eje X para Plotear Puntos", df)
            columnNameY = Parameters.SelectAxis(
                "Seleccione Eje Y para Plotear Puntos", df)

            if columnNameX != columnNameY:

                try:

                    X = df[columnNameX].values.reshape((-1, 1))
                    Y = df[columnNameY]

                    st.subheader("Resultados")
                 
                    fig, ax = plt.subplots(1,1)  #Cantidad de Graficas Filas y Columnas
                    fig.text(.5, -0.06, "Dispersion de Puntos, Segun las Columnas Seleccionadas", style = 'italic', fontsize= 8, ha='center', color = "blue")
                    ax.title.set_text('Grafica de Dispersion')
                    ax.set_xlabel(columnNameX)
                    ax.set_ylabel(columnNameY)
                    ax.scatter(X,Y)
                    
                    st.pyplot(fig)

                except:
                    st.error("No se ha podido Ejecutar La Operacion, Seleccione Otras Columnas y Vuelva a Intentar")
            
            else:
                st.warning("Las Columnas Deben De Ser Diferentes")

        if operation == 1:
            columnNameX = Parameters.SelectAxis("Seleccione Eje X para Definir Funcion de Tendencia", df)
            columnNameY = Parameters.SelectAxis("Seleccione Eje Y para Definir Funcion de Tendencia", df)

            if columnNameX != columnNameY:

                try:

                    X = df[columnNameX].values.reshape((-1, 1))
                    Y = df[columnNameY]

                    LinearRegression = linear_model.LinearRegression()
                    model = LinearRegression.fit(X,Y)
                    Y_PREDICT = model.predict(X)

                    #errores
                    errorMean = round(mean_squared_error(Y, Y_PREDICT, squared=True) , 4)
                    r2 =  round(r2_score(Y, Y_PREDICT) , 4)

                    st.subheader("Resultados")
                    modelStr = "y = {0}x {1} {2}".format(model.coef_[0], "+" if model.intercept_>0 else "",  model.intercept_)
                    
                    st.write("Modelo Lineal")
                    st.latex(modelStr)
                    
                    st.write("Errores del Modelo")
                    errorMeanStr = "Error Medio = {0}".format(errorMean)
                    r2Str = "R^2 = {0}".format(r2)

                    st.latex(errorMeanStr)
                    st.latex(r2Str)

                    st.write("Grafica de Tendencia")

                    fig, ax = plt.subplots(1,1)  #Cantidad de Graficas Filas y Columnas
                    fig.text(.5, -0.06, "Dispersion de Puntos, Con Tendencia", style = 'italic', fontsize= 8, ha='center', color = "blue")
                    ax.title.set_text('Grafica de Tendencia')
                    ax.set_xlabel(columnNameX)
                    ax.set_ylabel(columnNameY)
                    ax.scatter(X,Y)
                    ax.plot(X, Y_PREDICT, color="red")
                    
                    st.pyplot(fig)


                except:
                    st.error("No se ha podido Ejecutar La Operacion, Seleccione Otras Columnas y Vuelva a Intentar")
            
            else:
                st.warning("Las Columnas Deben De Ser Diferentes")

        if operation == 2:
            columnNameX = Parameters.SelectAxis("Seleccione Eje X para Realizar Prediccion del Modelo", df)
            columnNameY = Parameters.SelectAxis("Seleccione Eje Y para Realizar Prediccion del Modelo", df)
            numberPredict = Parameters.NumberInput()

            if columnNameX != columnNameY and numberPredict > 0:

                try:
                    X = df[columnNameX].values.reshape((-1, 1))
                    Y = df[columnNameY]

                    LinearRegression = linear_model.LinearRegression()
                    model = LinearRegression.fit(X,Y)
                    Y_PREDICT = model.predict(X)

                    #errores
                    errorMean = round(mean_squared_error(Y, Y_PREDICT, squared=True) , 4)
                    r2 =  round(r2_score(Y, Y_PREDICT) , 4)

                    #Prediccion
                    Y_NEW_PREDICTION = model.predict([[numberPredict]])

                    st.subheader("Resultados")
                    modelStr = "y = {0}x {1} {2}".format(model.coef_[0], "+" if model.intercept_>0 else "",  model.intercept_)
                    
                    st.write("Modelo Lineal")
                    st.latex(modelStr)
                    
                    st.write("Errores del Modelo")
                    errorMeanStr = "Error Medio = {0}".format(errorMean)
                    r2Str = "R^2 = {0}".format(r2)

                    st.latex(errorMeanStr)
                    st.latex(r2Str)

                    st.write("Prediccion")
                    predictionStr = "Prediccion = {0}".format(Y_NEW_PREDICTION[0])
                    st.latex(predictionStr)


                except:
                    st.error("No se ha podido Ejecutar La Operacion, Ingrese Nuevos Valores y Vuelva a Intentar")
            
            else:
                st.warning("Las Columnas Deben De Ser Diferentes y La Predicion Mayor a 0")



algoritmo()
