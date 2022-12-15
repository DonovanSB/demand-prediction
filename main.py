import streamlit as st
import pandas as pd
import numpy as np
from screens.dashboard import dashboard
from screens.data_upload import data_upload


st.sidebar.write("Menu")

menu = ["Cargar datos", "Dashboard"]
selected_option = st.sidebar.selectbox("Seleccione una opción", menu)

if selected_option == "Cargar datos":
    data_upload()
if selected_option == "Dashboard":
    dashboard()
