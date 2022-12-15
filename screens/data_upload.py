import streamlit as st
import pandas as pd

from services.data_service import DataService


service = DataService()


def data_upload():
    message = """
      Sube un archivo excel con los datos de potencia activa para la predicción
    """
    with st.container():
        st.header("Módulo de carga de datos")
        st.write(message)

        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            service.data_frame = df
