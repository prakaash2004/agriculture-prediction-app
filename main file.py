import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -------------------------------
# Title & Inputs
# -------------------------------
st.title("🌾 Agricultural Modal Price Forecasting")

state_input = st.text_input("Enter the State")
district_input = st.text_input("Enter the District")
commodity_input = st.text_input("Enter the Commodity")
future_year = st.number_input("Enter the future year (e.g., 2030):", min_value=2025, max_value=2100, step=1)

# -------------------------------
# Load agri.csv
# -------------------------------
@st.cache_data
def load_agri_data():
    try:
        agri_cols = ['state', 'district', 'commodity', 'arrival_date', 'modal_price']
        df = pd.read_csv("agri.csv", usecols=agri_cols)
        df['arrival_date'] = pd.to_datetime(df['arrival_date'], format="%d/%m/%Y", errors='coerce')
        df = df.dropna(subset=['arrival_date', 'modal_price'])
        df['year'] = df['arrival_date'].dt.year
        return df
    except Exception as e:
        st.error(f"🚫 Error loading agri.csv: {e}")
        return None

agri_df = load_agri_data()

# -------------------------------
# Load climate.csv
# -------------------------------
@st.cache_data
def load_climate_data():
    try:
        return pd.read_csv("climate.csv")
    except Exception as e:
        st.error(f"🚫 Error loading climate.csv: {e}")
        return None

climate_df = load_climate_data()

# -------------------------------
# Process & Forecast
# -------------------------------
if agri_df is not None and climate_df is not None and state_input and district_input and commodity_input:
    filtered = agri_df[
        (agri_df['state'] == state_input) &
        (
