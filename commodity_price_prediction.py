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

# =====================================
# Load Dataset
# =====================================
df = pd.read_csv('agrio.csv')

st.title("🌾 Agri Commodity Price & Rainfall Based Forecasting")

# Sidebar for user selections
st.sidebar.header("Select Your Options")
commodity = st.sidebar.selectbox("Select Commodity", sorted(df['Commodity'].unique()))

if commodity:
    states = df[df['Commodity'] == commodity]['State'].unique()
    state = st.sidebar.selectbox("Select State", sorted(states))

    if state:
        districts = df[(df['Commodity'] == commodity) & (df['State'] == state)]['District'].unique()
        district = st.sidebar.selectbox("Select District", sorted(districts))

        if district:
            st.subheader(f"Data for {commodity} in {district}, {state}")

            # Filter Data
            df_filtered = df[
                (df['Commodity'] == commodity) &
                (df['State'] == state) &
                (df['District'] == district)
            ]

            if df_filtered.empty:
                st.error("No matching data found.")
            else:
                st.write("### 📈 Historical Data")
                st.dataframe(df_filtered[['Year', 'Market', 'Price per kg (INR)', 'Rainfall (cm)']])

                future_year = st.number_input("Select Future Year", min_value=int(df_filtered['Year'].max()) + 1, max_value=2100, value=2030)

                if st.button("Run Prediction"):
                    # Aggregate by Year
                    df_agg = df_filtered.groupby('Year').agg({
                        'Price per kg (INR)': 'mean',
                        'Rainfall (cm)': 'mean'
                    }).reset_index()

                    df_agg.rename(columns={'Price per kg (INR)': 'modal_price'}, inplace=True)

                    data_for_model = df_agg[['Year', 'modal_price', 'Rainfall (cm)']]

                    # Scaling
                    scaler = MinMaxScaler()
                    scaled_data = scaler.fit_transform(data_for_model)

                    # Create Sequences
                    def create_sequences(data, look_back=3):
                        X, y = [], []
                        for i in range(len(data) - look_back):
                            X.append(data[i:i+look_back])
                            y.append(data[i+look_back, 1])  # Predict modal_price
                        return np.array(X), np.array(y)

                    look_back = 3
                    X_seq, y_seq = create_sequences(scaled_data)

                    # Build LSTM
                    model = Sequential([
                        Input(shape=(look_back, scaled_data.shape[1])),
                        LSTM(60, activation='relu'),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mse')
                    model.fit(X_seq, y_seq, epochs=80, verbose=0)

                    # Predict Future
                    last_year = int(df_agg['Year'].max())
                    n_future = future_year - last_year

                    current_seq = scaled_data[-look_back:].copy()
                    predicted_years = []
                    predicted_prices_scaled = []

                    rainfall_base = df_agg['Rainfall (cm)'].iloc[-1]

                    for _ in range(n_future):
                        pred_scaled = model.predict(current_seq[np.newaxis, :])[0][0]
                        predicted_prices_scaled.append(pred_scaled)
                        predicted_years.append(last_year + 1)

                        # Add slight random variation and rainfall adjustment
                        rainfall_adjustment = np.random.normal(0, 0.01)
                        random_noise = np.random.normal(0, 0.015)

                        next_year_rainfall = rainfall_base + np.random.normal(0, 2)
                        next_year_rainfall_scaled = scaler.transform([[last_year + 1, pred_scaled + random_noise, next_year_rainfall]])[0][2]

                        next_row = np.array([last_year + 1, pred_scaled + random_noise, next_year_rainfall_scaled])

                        current_seq = np.vstack([current_seq[1:], next_row])
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

                    # Plotting
                    fig, ax = plt.subplots(figsize=(12,6))
                    ax.plot(df_agg['Year'], historical_prices, marker='o', label='Historical Price')
                    ax.plot(predicted_years, future_prices, marker='x', linestyle='--', color='red', label='Predicted Price')
                    ax.set_xlabel("Year")
                    ax.set_ylabel("Modal Price (INR)")
                    ax.set_title(f"{commodity} Price Forecast in {district}, {state}")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

                    # Final Predicted Price
                    st.success(f"📌 Predicted Modal Price of {commodity} in {district}, {state} for {future_year} is ₹{future_prices[-1]:.2f}/kg")
