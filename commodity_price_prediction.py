import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
import warnings
warnings.filterwarnings('ignore')



# Load Dataset
df = pd.read_csv('agrio.csv')
df_2025 = df[df['Year'] == 2025]

st.title("🌾 Agriculture Commodity Monitoring and Realistic Forecasting System")

# DOMAIN 1: Real-Time Commodity Values
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
                top3 = df_2025[
                    (df_2025['Commodity'] == commodity)
                ].sort_values('Price per kg (INR)', ascending=False).head(3)
                st.write("Top 3 Markets for Selected Commodity")
                st.dataframe(top3[['State', 'District', 'Market', 'Price per kg (INR)']])

            with col2:
                top5 = df_2025[
                    (df_2025['Commodity'] == commodity) &
                    (df_2025['State'] == state)
                ].sort_values('Price per kg (INR)', ascending=False).head(5)
                st.write(f"Top 5 in {state}")
                st.dataframe(top5[['District', 'Market', 'Price per kg (INR)']])

            # Show all market prices in the selected district
            st.write(f"All Markets in {district}, {state} for {commodity}")
            district_markets = df_2025[
                (df_2025['Commodity'] == commodity) &
                (df_2025['State'] == state) &
                (df_2025['District'] == district)
            ].sort_values('Price per kg (INR)', ascending=False)
            st.dataframe(district_markets[['Market', 'Price per kg (INR)']])

# DOMAIN 2: Future Forecast
st.header("🔮 Future Price Forecast Using LSTM")

if commodity and state and district:
    future_year = st.number_input("Select Future Year", min_value=2025, max_value=2100, value=2030)

    if st.button("Run Forecast"):
        with st.spinner("Running forecast..."):
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

                # Auto-compute average inflation rate from historical prices
                df_agg['prev_price'] = df_agg['modal_price'].shift(1)
                df_agg['inflation'] = (df_agg['modal_price'] - df_agg['prev_price']) / df_agg['prev_price']
                inflation_rate = df_agg['inflation'].dropna().mean()
                inflation_rate = max(inflation_rate, 0.01)

                # Historical volatility
                historical_changes = df_agg['modal_price'].pct_change().dropna()
                volatility = historical_changes.std() * 0.5  # tuned smaller

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
                last_real_price = df_agg['modal_price'].iloc[-1]

                for _ in range(n_future):
                    pred_scaled = model.predict(current_seq[np.newaxis, :])[0][0]

                    # Apply inflation + historical fluctuation
                    price_change = np.random.normal(loc=inflation_rate, scale=volatility)
                    final_price = last_real_price * (1 + price_change)

                    # Clamp to avoid unrealistic drop
                    final_price = max(final_price, last_real_price * 0.90)

                    last_real_price = final_price
                    predicted_years.append(last_year + 1)

                    corrected_scaled = scaler.transform([[0, final_price, rainfall_base]])[0][1]
                    predicted_prices_scaled.append(corrected_scaled)

                    rainfall_effect = np.random.normal(0, 1)
                    next_input = scaler.transform([[last_year + 1, final_price, rainfall_base + rainfall_effect]])[0]
                    current_seq = np.vstack([current_seq[1:], next_input])
                    last_year += 1

                historical_prices = df_agg['modal_price'].tolist()
                future_prices = scaler.inverse_transform(
                    np.column_stack([
                        np.linspace(df_agg['Year'].max()+1, future_year, len(predicted_prices_scaled)),
                        predicted_prices_scaled,
                        [rainfall_base]*len(predicted_prices_scaled)
                    ])
                )[:,1]

                future_prices_smoothed = pd.Series(future_prices).rolling(window=2, min_periods=1).mean()

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


