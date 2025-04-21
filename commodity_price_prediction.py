# commodity_price_prediction.py (Simplified and Corrected Version)

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Commodity Price Predictor', layout='wide')
st.title('📈 Commodity Price Prediction App')

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('agrio.csv')

data = load_data()

# Sidebar for user selection
st.sidebar.header('Select Options')
commodity = st.sidebar.selectbox('Commodity:', sorted(data['Commodity'].unique()))
state = st.sidebar.selectbox('State:', sorted(data[data['Commodity'] == commodity]['State'].unique()))
district = st.sidebar.selectbox('District:', sorted(data[(data['Commodity'] == commodity) & (data['State'] == state)]['District'].unique()))

# Filter data
filtered_data = data[(data['Commodity'] == commodity) & (data['State'] == state) & (data['District'] == district)]

if filtered_data.empty:
    st.error("No data available for the selected combination.")
else:
    df_agg = filtered_data.groupby('Year')['Price per kg (INR)'].mean().reset_index()

    # Plot historical prices
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_agg['Year'], df_agg['Price per kg (INR)'], marker='o', color='blue', label='Historical Prices')
    ax.set_xlabel('Year')
    ax.set_ylabel('Price per kg (INR)')
    ax.set_title(f'{commodity} Prices in {district}, {state}')
    ax.grid()
    ax.legend()

    st.pyplot(fig)

    # Display data table
    st.subheader('Historical Price Data')
    st.dataframe(df_agg)

    # Simple future price estimation
    avg_growth_rate = df_agg['Price per kg (INR)'].pct_change().mean()
    last_price = df_agg['Price per kg (INR)'].iloc[-1]
    future_year = st.sidebar.number_input('Predict for year:', min_value=df_agg['Year'].max() + 1, max_value=2100, value=df_agg['Year'].max() + 1)

    years_to_predict = future_year - df_agg['Year'].max()
    predicted_price = last_price * ((1 + avg_growth_rate) ** years_to_predict)

    st.success(f'✅ Predicted price of {commodity} in {district}, {state} for the year {future_year} is ₹{predicted_price:.2f}/kg.')
