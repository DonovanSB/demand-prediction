import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from os import path, pardir

from services.data_service import DataService
from services.mode_service import LstmModel, StockDataset, calculate_metrics


service = DataService()


def on_click():
    global container
    data = service.df
    # data[data < 0] = 0
    df1 = data.reset_index()["TkW"]

    batch_size = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scalar = MinMaxScaler(feature_range=(0, 1))
    df1 = scalar.fit_transform(np.array(df1).reshape(-1, 1))

    seq_len = 100
    test_dataset = StockDataset(df1, seq_len)
    data_loader = DataLoader(test_dataset, batch_size, drop_last=True)

    input_dim = 1
    hidden_size = 10
    num_layers = 2

    model = LstmModel(input_dim, hidden_size, num_layers).to(device)
    parent = path.join(Path(__file__).parent.resolve(), pardir)
    root = path.abspath(parent)
    model.load_state_dict(torch.load(path.join(root, "models/demand_prediction.pt")))
    model.to(device)

    metric, pred_arr, y_arr = calculate_metrics(model, scalar, data_loader)
    with container:
        st.title("Predicciones")
        st.write("MSE", metric)
        st.line_chart(pd.DataFrame({"prediction": pred_arr, "real": y_arr}))


def dashboard():
    global container
    with st.container():
        st.header("Dashboard")
        if service.df is not None:
            st.write("Datos cargados")
            st.line_chart(service.df.TkW)
            st.button("Realizar predicción", on_click=on_click)
            [container] = st.columns(1)
        else:
            st.write("Por favor carga los datos")
