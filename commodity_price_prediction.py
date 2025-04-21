# commodity_price_prediction.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

st.set_page_config(page_title='Commodity Price Predictor', layout='wide')
st.title('📈 Commodity Price Prediction App')

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('agrio.csv')

data = load_data()
data_2025 = data[data['Year'] == 2025]

# Sidebar for selections
st.sidebar.header('Select Options')

commodity = st.sidebar.selectbox('Commodity:', sorted(data_2025['Commodity'].unique()))
state_options = sorted(data_2025[data_2025['Commodity'] == commodity]['State'].unique())
state = st.sidebar.selectbox('State:', state_options)

district_options = sorted(data_2025[(data_2025['Commodity'] == commodity) & (data_2025['State'] == state)]['District'].unique())
district = st.sidebar.selectbox('District:', district_options)

future_year = st.sidebar.number_input('Predict up to year:', min_value=2026, max_value=2100, value=2030)

# Main page
st.subheader('Selected Details')
st.write(f"**Commodity:** {commodity} | **State:** {state} | **District:** {district} | **Predict Until:** {future_year}")

# Filter data for visualization
filtered_data = data[(data['Commodity'] == commodity) & (data['State'] == state) & (data['District'] == district)]

if filtered_data.empty:
    st.error("No data available for the selected combination.")
else:
    df_agg = filtered_data.groupby('Year')['Price per kg (INR)'].mean().reset_index()
    df_agg.rename(columns={'Price per kg (INR)': 'modal_price'}, inplace=True)

    # Prepare data for model
    scaler_X = MinMaxScaler()
    scaled_data = scaler_X.fit_transform(df_agg.values)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(df_agg['modal_price'].values.reshape(-1, 1))

    def create_sequences(data, target, look_back=3):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i+look_back])
            y.append(target[i+look_back])
        return np.array(X), np.array(y)

    look_back = 3
    X_seq, y_seq = create_sequences(scaled_data, y_scaled, look_back)

    # LSTM Model
    model = Sequential([
        Input(shape=(look_back, scaled_data.shape[1])),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_seq, y_seq, epochs=50, verbose=0)

    # Prediction
    last_year = df_agg['Year'].max()
    n_future = future_year - last_year

    current_seq = scaled_data[-look_back:].copy()
    predicted_years = []
    predicted_prices = []

    for i in range(n_future):
        pred = model.predict(current_seq[np.newaxis, :])[0]
        raw_pred = scaler_y.inverse_transform(pred.reshape(-1, 1))[0][0]

        raw_pred = max(raw_pred, predicted_prices[-1]) if predicted_prices else max(raw_pred, df_agg['modal_price'].iloc[-1])

        new_year = last_year + i + 1
        predicted_years.append(new_year)
        predicted_prices.append(raw_pred)

        new_row_raw = np.array([new_year, raw_pred])
        new_row_scaled = scaler_X.transform(new_row_raw.reshape(1, -1))[0]
        current_seq = np.vstack([current_seq[1:], new_row_scaled])

    # Smooth predictions
    smoothed_prices = [predicted_prices[0]]
    for i in range(1, len(predicted_prices)):
        curr_price = predicted_prices[i]
        prev_price = smoothed_prices[-1]
        if curr_price < prev_price * 0.97:
            curr_price = prev_price * 0.97
        smoothed_prices.append(curr_price)
    predicted_prices = smoothed_prices

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_agg['Year'], df_agg['modal_price'], marker='o', label='Historical')
    ax.plot(predicted_years, predicted_prices, marker='x', linestyle='--', color='red', label='Predicted')
    ax.set_xlabel('Year')
    ax.set_ylabel('Price (INR/kg)')
    ax.set_title(f'Price Forecast for {commodity} in {district}, {state}')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # Heatmap
    all_years = list(df_agg['Year']) + predicted_years
    all_prices = list(df_agg['modal_price']) + predicted_prices
    heat_df = pd.DataFrame({'Year': all_years, 'Price': all_prices})
    heat_df['Type'] = ['Historical']*len(df_agg) + ['Predicted']*len(predicted_years)

    pivot = heat_df.pivot_table(index='Type', columns='Year', values='Price')

    fig2, ax2 = plt.subplots(figsize=(12, 3))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    # Final Prediction
    st.success(f'✅ Predicted modal price of {commodity} in {district}, {state} for the year {future_year} is ₹{predicted_prices[-1]:.2f}/kg.')