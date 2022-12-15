import streamlit as st
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from services.data_service import DataService


service = DataService()


def on_click():
    data = service.df
    # data[data < 0] = 0
    df1 = data.reset_index()["TkW"]
    st.line_chart(df1)

    scalar = MinMaxScaler(feature_range=(0, 1))
    df1 = scalar.fit_transform(np.array(df1).reshape(-1, 1))

    batch_size = 100
    data_loader = DataLoader(df1, batch_size, drop_last=True)
    st.write(len(data_loader))


def dashboard():
    message = """
      Sube un archivo excel con los datos de potencia activa para la predicción
    """
    with st.container():
        st.header("Dashboard")
        if service.df is not None:
            st.write("Datos cargados")
            st.line_chart(service.df.TkW)
            st.button("Realizar predicción", on_click=on_click)
        else:
            st.write("Por favor carga los datos")
