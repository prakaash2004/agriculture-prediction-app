import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_csv('agrio.csv')
df_2025 = df[df['Year'] == 2025]

st.title("📊 Agri Commodity Explorer & Forecasting with Rainfall")

# ============================================
# DOMAIN 1: Real-Time 2025 Commodity Value
# ============================================

st.header("🔍 Explore Real-Time Commodity Values (2025)")

commodity = st.selectbox("Select Commodity", sorted(df_2025['Commodity'].unique()))
if commodity:
    filtered_states = df_2025[df_2025['Commodity'] == commodity]['State'].unique()
    state = st.selectbox("Select State", sorted(filtered_states))

    if state:
        filtered_districts = df_2025[
            (df_2025['Commodity'] == commodity) &
            (df_2025['State'] == state)
        ]['District'].unique()
        district = st.selectbox("Select District", sorted(filtered_districts))

        if district:
            st.subheader("🔝 Top 3 Markets by Price (2025)")
            top3 = df_2025[df_2025['Commodity'] == commodity].sort_values('Price per kg (INR)', ascending=False).head(3)
            st.dataframe(top3[['State', 'District', 'Market', 'Price per kg (INR)']])

            st.subheader("🏆 Top 5 Prices in Selected State")
            top5 = df_2025[
                (df_2025['Commodity'] == commodity) &
                (df_2025['State'] == state)
            ].sort_values('Price per kg (INR)', ascending=False).head(5)
            st.dataframe(top5[['District', 'Market', 'Price per kg (INR)']])

            st.subheader("📍 Markets in Selected District")
            market_df = df_2025[
                (df_2025['Commodity'] == commodity) &
                (df_2025['State'] == state) &
                (df_2025['District'] == district)
            ].sort_values('Price per kg (INR)', ascending=False)
            st.dataframe(market_df[['Market', 'Price per kg (INR)']])

# ============================================
# DOMAIN 2: LSTM FUTURE PRICE PREDICTION
# ============================================

st.header("📈 Future Price Forecast Using LSTM (With Rainfall)")

if commodity and state and district:
    future_year = st.number_input("Select Future Year", min_value=2025, max_value=2100, value=2030)

    if st.button("Run LSTM Prediction"):
        df_filtered = df[
            (df['State'] == state) &
            (df['District'] == district) &
            (df['Commodity'] == commodity)
        ]

        if df_filtered.empty:
            st.error("No matching data found.")
        else:
            df_agg = df_filtered.groupby('Year').agg({
                'Price per kg (INR)': 'mean',
                'Rainfall (cm)': 'mean'
            }).reset_index()

            df_agg.rename(columns={'Price per kg (INR)': 'modal_price'}, inplace=True)

            # Prepare input features: Year, Modal Price, Rainfall
            features = df_agg[['Year', 'modal_price', 'Rainfall (cm)']]

            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(features)

            def create_sequences(data, look_back=3):
                X, y = [], []
                for i in range(len(data) - look_back):
                    X.append(data[i:i+look_back])
                    y.append(data[i+look_back, 1])  # Predict Modal Price
                return np.array(X), np.array(y)

            look_back = 3
            X_seq, y_seq = create_sequences(scaled_features)

            # Build and Train LSTM
            model = Sequential([
                Input(shape=(look_back, scaled_features.shape[1])),
                LSTM(60, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_seq, y_seq, epochs=80, verbose=0)

            # Predict Future
            last_year = df_agg['Year'].max()
            n_future = future_year - last_year

            current_seq = scaled_features[-look_back:].copy()
            predicted_years = []
            predicted_prices_scaled = []

            rainfall_base = df_agg['Rainfall (cm)'].iloc[-1]

            for _ in range(n_future):
                pred_scaled = model.predict(current_seq[np.newaxis, :])[0][0]
                predicted_prices_scaled.append(pred_scaled)
                predicted_years.append(last_year + 1)

                # Add slight variation and rainfall adjustment
                random_variation = np.random.normal(0, 0.015)
                next_rainfall = rainfall_base + np.random.normal(0, 2)

                next_input = scaler.transform([[last_year + 1, pred_scaled + random_variation, next_rainfall]])[0]
                current_seq = np.vstack([current_seq[1:], next_input])
                last_year += 1

            # Inverse scale to get real prices
            historical_prices = df_agg['modal_price'].tolist()
            future_prices = scaler.inverse_transform(
                np.column_stack([
                    np.linspace(df_agg['Year'].max()+1, future_year, len(predicted_prices_scaled)),
                    predicted_prices_scaled,
                    [rainfall_base]*len(predicted_prices_scaled)
                ])
            )[:,1]

            # Plotting
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(df_agg['Year'], historical_prices, marker='o', label='Historical Price')
            ax.plot(predicted_years, future_prices, marker='x', linestyle='--', color='red', label='Predicted Price')
            ax.set_xlabel("Year")
            ax.set_ylabel("Price (INR)")
            ax.set_title(f"{commodity} Price Forecast in {district}, {state}")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Show Result
            st.success(f"📌 Predicted modal price of {commodity} in {district}, {state} for {future_year} is ₹{future_prices[-1]:.2f}/kg")
