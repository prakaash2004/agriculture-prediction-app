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

# Load Dataset
df = pd.read_csv('agrio.csv')
df_2025 = df[df['Year'] == 2025]

st.title("🌾 Agriculture Commodity Monitoring and Realistic Forecasting System")

# -------------------- DOMAIN 1: Real-Time Commodity Explorer --------------------
st.header("📊 Real-Time 2025 Commodity Explorer")
commodity = st.selectbox("Select Commodity", sorted(df_2025['Commodity'].unique()))

if commodity:
    st.subheader(f"🌟 Top 3 Prices in India for {commodity}")
    top3_india = df_2025[df_2025['Commodity'] == commodity].sort_values('Price per kg (INR)', ascending=False).head(3)
    st.dataframe(top3_india[['State', 'District', 'Market', 'Price per kg (INR)']])

    states = df_2025[df_2025['Commodity'] == commodity]['State'].unique()
    state = st.selectbox("Select State", sorted(states))

    if state:
        st.subheader(f"🏆 Top 3 Prices in {state} for {commodity}")
        top3_state = df_2025[
            (df_2025['Commodity'] == commodity) &
            (df_2025['State'] == state)
        ].sort_values('Price per kg (INR)', ascending=False).head(3)
        st.dataframe(top3_state[['District', 'Market', 'Price per kg (INR)']])

        districts = df_2025[
            (df_2025['Commodity'] == commodity) &
            (df_2025['State'] == state)
        ]['District'].unique()
        district = st.selectbox("Select District", sorted(districts))

        if district:
            st.subheader(f"🛒 All {commodity} Market Prices in {district}, {state}")
            all_markets = df_2025[
                (df_2025['Commodity'] == commodity) &
                (df_2025['State'] == state) &
                (df_2025['District'] == district)
            ].sort_values('Price per kg (INR)', ascending=False)
            st.dataframe(all_markets[['Market', 'Price per kg (INR)']])

# -------------------- DOMAIN 2: LSTM Forecasting --------------------
st.header("🔮 Future Price Forecast Using LSTM (Realistic 2-year fall limit)")

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
            fall_years = 0
            previous_price = df_agg['modal_price'].iloc[-1]
            historical_prices = df_agg['modal_price'].tolist()

            # Average inflation rate from real data
            past_prices = df_agg['modal_price'].tolist()
            inflation_rates = [
                (past_prices[i] - past_prices[i-1]) / past_prices[i-1]
                for i in range(1, len(past_prices))
                if past_prices[i-1] > 0
            ]
            avg_inflation = np.mean(inflation_rates)

            for _ in range(n_future):
                pred_scaled = model.predict(current_seq[np.newaxis, :])[0][0]

                yearly_inflation = np.random.normal(loc=avg_inflation, scale=0.01)
                random_fluctuation = pred_scaled * np.random.normal(0, 0.005)
                rainfall_effect = np.random.normal(0, 1)

                corrected_pred_scaled = pred_scaled + (pred_scaled * yearly_inflation) + random_fluctuation

                real_predicted_price = scaler.inverse_transform(
                    np.array([[0, corrected_pred_scaled, rainfall_base]])
                )[0][1]

                if real_predicted_price < previous_price:
                    fall_years += 1
                else:
                    fall_years = 0

                if fall_years > 2:
                    real_predicted_price = previous_price * 1.02
                    fall_years = 0

                previous_price = real_predicted_price

                corrected_pred_scaled = scaler.transform(
                    np.array([[0, real_predicted_price, rainfall_base]])
                )[0][1]

                predicted_prices_scaled.append(corrected_pred_scaled)
                predicted_years.append(last_year + 1)

                next_input = scaler.transform([[last_year + 1, corrected_pred_scaled, rainfall_base + rainfall_effect]])[0]
                current_seq = np.vstack([current_seq[1:], next_input])
                last_year += 1

            # Inverse transform predicted
            future_prices = scaler.inverse_transform(
                np.column_stack([
                    np.linspace(df_agg['Year'].max()+1, future_year, len(predicted_prices_scaled)),
                    predicted_prices_scaled,
                    [rainfall_base]*len(predicted_prices_scaled)
                ])
            )[:,1]

            future_prices_smoothed = pd.Series(future_prices).rolling(window=2, min_periods=1).mean()

            # Plot
            st.subheader("📈 Final Corrected Price Prediction Graph")
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(df_agg['Year'], historical_prices, marker='o', label='Historical Price', color='blue')
            ax.plot(predicted_years, future_prices_smoothed, marker='x', linestyle='--', color='green', label='Predicted Price')
            ax.set_xlabel("Year")
            ax.set_ylabel("Price (INR)")
            ax.set_title(f"{commodity} Price Forecast for {district}, {state}")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            st.success(f"📌 Predicted Modal Price of {commodity} in {district}, {state} for {future_year} is ₹{future_prices_smoothed.iloc[-1]:.2f}/kg")
