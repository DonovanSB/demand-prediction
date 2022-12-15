import streamlit as st
import pandas as pd

from services.data_service import DataService


service = DataService()


def dashboard():
    message = """
      Sube un archivo excel con los datos de potencia activa para la predicción
    """
    with st.container():
        st.header("Dashboard")
        if service.df is not None:
            st.write("Datos cargados")
            st.line_chart(service.df.TkW)
        else:
            st.write("Por favor carga los datos")
