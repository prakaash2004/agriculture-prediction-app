import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
import warnings
warnings.filterwarnings('ignore')

# Page Setup
st.set_page_config(page_title="Agri Price Forecast", page_icon="🌾", layout="wide")

# Load dataset
df = pd.read_csv('agrio.csv')
df_2025 = df[df['Year'] == 2025]

st.title("🌾 Agri Commodity Monitoring and Forecasting System")

# DOMAIN 1: Real-time Commodity Values
st.header("📊 Real-Time 2025 Commodity Explorer")
commodity = st.selectbox("Select Commodity", sorted(df_2025['Commodity'].unique()))

if commodity:
    states = df_2025[df_2025['Commodity'] == commodity]['State'].unique()
    state = st.selectbox("Select State", sorted(states))

    if state:
        districts = df_2025[
            (df_2025['Commodity'] == commodity) &
            (df_2025['State'] == state)
        ]['District'].unique()
        district = st.selectbox("Select District", sorted(districts))

        if district:
            st.subheader(f"🔍 {commodity} Markets in {district}, {state}")

            col1, col2 = st.columns(2)
            with col1:
                top3 = df_2025[df_2025['Commodity'] == commodity].sort_values('Price per kg (INR)', ascending=False).head(3)
                st.write("Top 3 Markets")
                st.dataframe(top3[['State', 'District', 'Market', 'Price per kg (INR)']])

            with col2:
                top5 = df_2025[
                    (df_2025['Commodity'] == commodity) &
                    (df_2025['State'] == state)
                ].sort_values('Price per kg (INR)', ascending=False).head(5)
                st.write(f"Top 5 in {state}")
                st.dataframe(top5[['District', 'Market', 'Price per kg (INR)']])

# DOMAIN 2: Future Forecast
st.header("🔮 Future Price Forecast Using LSTM + Inflation")

if commodity and state and district:
    future_year = st.number_input("Select Future Year", min_value=2025, max_value=2100, value=2030)

    if st.button("Run Forecast"):
        df_filtered = df[
            (df['State'] == state) &
            (df['District'] == district) &
            (df['Commodity'] == commodity)
        ]

        if df_filtered.empty:
            st.error("No data found.")
        else:
            df_agg = df_filtered.groupby('Year').agg({
                'Price per kg (INR)': 'mean',
                'Rainfall (cm)': 'mean'
            }).reset_index()

            df_agg.rename(columns={'Price per kg (INR)': 'modal_price'}, inplace=True)

            features = df_agg[['Year', 'modal_price', 'Rainfall (cm)']]
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(features)

            def create_sequences(data, look_back=3):
                X, y = [], []
                for i in range(len(data) - look_back):
                    X.append(data[i:i+look_back])
                    y.append(data[i+look_back, 1])
                return np.array(X), np.array(y)

            look_back = 3
            X_seq, y_seq = create_sequences(scaled)

            model = Sequential([
                Input(shape=(look_back, scaled.shape[1])),
                LSTM(80, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_seq, y_seq, epochs=100, verbose=0)

            last_year = df_agg['Year'].max()
            n_future = future_year - last_year

            current_seq = scaled[-look_back:].copy()
            predicted_years = []
            predicted_prices_scaled = []

            rainfall_base = df_agg['Rainfall (cm)'].iloc[-1]

            for _ in range(n_future):
                pred_scaled = model.predict(current_seq[np.newaxis, :])[0][0]

                # Add +2% inflation growth
                inflation_adjustment = pred_scaled * 0.02

                # Mild random noise
                random_variation = np.random.normal(0, 0.015)

                # Rainfall impact
                rainfall_effect = np.random.normal(0, 2)

                pred_scaled = pred_scaled + inflation_adjustment + random_variation

                predicted_prices_scaled.append(pred_scaled)
                predicted_years.append(last_year + 1)

                next_input = scaler.transform([[last_year + 1, pred_scaled, rainfall_base + rainfall_effect]])[0]
                current_seq = np.vstack([current_seq[1:], next_input])
                last_year += 1

            # Inverse scaling
            historical_prices = df_agg['modal_price'].tolist()
            future_prices = scaler.inverse_transform(
                np.column_stack([
                    np.linspace(df_agg['Year'].max()+1, future_year, len(predicted_prices_scaled)),
                    predicted_prices_scaled,
                    [rainfall_base]*len(predicted_prices_scaled)
                ])
            )[:,1]

            # Moving Average Smoothing
            future_prices_smoothed = pd.Series(future_prices).rolling(window=2, min_periods=1).mean()

            # Plot
            st.subheader("📈 Price Prediction Graph")
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(df_agg['Year'], historical_prices, marker='o', label='Historical', color='blue')
            ax.plot(predicted_years, future_prices_smoothed, marker='x', linestyle='--', color='green', label='Predicted')
            ax.set_xlabel("Year")
            ax.set_ylabel("Price (INR)")
            ax.set_title(f"{commodity} Price Forecast in {district}, {state}")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            st.success(f"📌 Predicted Modal Price of {commodity} in {district}, {state} for {future_year} is ₹{future_prices_smoothed.iloc[-1]:.2f}/kg")
