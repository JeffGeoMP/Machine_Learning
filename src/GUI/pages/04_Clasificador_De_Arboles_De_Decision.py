import streamlit as st
from PIL import Image
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np


import Helpers.HandlerFiles as Files
import Helpers.HandlerParameters as Parameters


def information():
    st.sidebar.write("## Arbol de Decisión 📑")

    st.sidebar.write("¿Qué es?")
    st.sidebar.write("El árbol de decisiones, se trata de un diagrama de flujo que empieza con una idea principal y luego se ramifica según las consecuencias " + 
                    "de tus decisiones. Se denomina `árbol de decisiones` por su semejanza con un árbol con muchas ramas.")

    st.sidebar.write("Este método se usa para realizar un análisis que consiste en delinear de forma gráfica los posibles resultados, costos y consecuencias de " +
                    "una decisión compleja. Se puede usar un árbol de decisiones para calcular el valor esperado de cada resultado en función de las decisiones tomadas " + 
                    "y sus respectivas consecuencias. Luego, se puede comparar los diferentes resultados para determinar rápidamente cuál será el mejor plan de acción. " + 
                    "También se puede utilizar un árbol de decisiones para resolver problemas, gestionar costos e identificar oportunidades.​")


def algoritmo():
    information()

    st.title("Algoritmo - Árbol de Decisiones")

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

                # Comenzammos el algoritmo del arbol
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

                # Creacion y entrenamiendo del modelo
                clf = DecisionTreeClassifier().fit(features, encodedClass)
                

                #Prediccion 
                prediction = clf.predict([encodedPredict])
                predictClass = le.inverse_transform(prediction)

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

                st.write("Árbol de Decisión")
                fig, ax = plt.subplots(figsize=(12,12))
                fig.suptitle('Árbol de Decisiones', fontsize="20")
                fig.text(.5, -0.025, "Decisiones, Según las Columnas Seleccionadas", style = 'italic', fontsize= 15, ha='center', color = "red")
                plot_tree(clf, filled=True)
                plt.savefig("assets/tree.png")
                plt.close()

                image = Image.open("assets/tree.png")
                st.image(image, caption='Correlacion de Puntos, Sobre las Columnas Seleccioandas')

            except Exception as e:
                st.warning("No Se A Podido Ejecutar La Operación, Seleccione Nuevos Parametros y Vuelva a Intentar")
                print(e)


algoritmo()
