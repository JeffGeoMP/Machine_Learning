from traceback import print_tb
import streamlit as st
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import plot_tree
import numpy as np


import Helpers.HandlerFiles as Files
import Helpers.HandlerParameters as Parameters


def information():
    st.sidebar.write("## Clasificador Gaussiano 📑")

    st.sidebar.write("¿Qué es?")
    st.sidebar.write(
        "El Clasificador Gaussiano es uno de los algoritmos más simples y poderosos para la clasificación basado en el Teorema de Bayes con una suposición " +
        "de independencia entre los predictores. El Clasificador Gaussiano es fácil de construir y particularmente útil para conjuntos de datos muy grandes.")

    st.sidebar.write("El Clasificador Gaussiano asume que el efecto de una característica particular en una clase es independiente de otras características. " +
                     "Por ejemplo, un solicitante de préstamo es deseable o no dependiendo de sus ingresos, historial de préstamos y transacciones anteriores, edad y ubicación. " +
                     "Incluso si estas características son interdependientes, estas características se consideran de forma independiente. Esta suposición simplifica la computación, " +
                     "y por eso se considera ingenua. Esta suposición se denomina independencia condicional de clase.​")


def algoritmo():
    information()

    st.title("Algoritmo - Clasificador Gaussiano")

    st.subheader("Carga de Archivo")
    df = Files.uploadFile("Seleccione el Archivo")

    if df is not None:
        st.write("Información Cargada")
        st.dataframe(df)

        # Pedimos los parametros
        st.markdown("""---""")
        st.subheader("Parámetros")
        columnsClassifiquer = Parameters.SelectColumns("Seleccione Las Columnas Para Clasificar", df)
        columnClass = Parameters.SelectAxis("Seleccione Columna Clase", df)

        if len(columnsClassifiquer) > 0:
            try:
                st.write("Seleccione Valores para Predición, Según las Columnas Seleccionadas")
                cols = st.columns(len(columnsClassifiquer))
                predictValues = []
                count = 0
                for i in columnsClassifiquer:
                    col = cols[count]
                    predictValues.append(col.selectbox(str(i), df[i].unique(), 0))
                    count = count + 1

                # Comenzammos el algoritmo de Gauss
                le = preprocessing.LabelEncoder()

                # Codificamos valores para las columnas seleccionadas y los valores de prediccion
                encodedList = []
                encodedPredict = []
                count = 0
                for i in columnsClassifiquer:
                    encodedList.append(le.fit_transform(df[i]))
                    encodedPredict.append(np.where(le.classes_ == predictValues[count])[0][0])  # buscamos clase para prediccion
                    count = count + 1

                #Hacemos tuplas con la codificacion anterior
                features = list(zip(*encodedList))

                # Codificamos columna Clase
                encodedClass = le.fit_transform(df[columnClass])

                # Creacion y entrenamiendo del modelo Gaussiano
                model = GaussianNB()
                model.fit(features, encodedClass)

                predict = model.predict([encodedPredict])
                predictClass = le.inverse_transform(predict)

                #Resultados
                st.markdown("""---""")
                st.subheader("Resultados")
                st.write("Matriz Codificada para Columnas de Clasificación Seleccionadas")
                st.dataframe(features)

                st.write("Valores de Predicción Codificados")
                predictStr = ""
                countTemp = 0
                for i in predictValues:
                    predictStr += "{0} \\rightarrow {1} \\newline \n".format(i, encodedPredict[countTemp])
                    countTemp = countTemp + 1
                
                st.latex(predictStr)

                st.write("Prediccion")
                predictionStr= "Prediccion \\rightarrow {0}".format(predictClass[0])
                st.latex(predictionStr)

            except Exception as e:
                st.warning("No Se A Podido Ejecutar La Operación, Seleccione Nuevos Parametros y Vuelva a Intentar")
                print(e)


algoritmo()
