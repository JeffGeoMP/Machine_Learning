import sys
from os.path import dirname, abspath

path = dirname(dirname(dirname(abspath(__file__)))) + "\Helpers"
sys.path.append(path)

import streamlit as st
import HandlerFiles as Files


st.markdown("# Regresion Lineal")

Files.uploadFile()

