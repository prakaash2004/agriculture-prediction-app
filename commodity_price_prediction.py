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

# ==============================
# Load dataset
# ==============================
df = pd.read_csv('agrio.csv')

st.title("🌾 Agriculture Commodity Price & Rainfall Prediction System")

# User Inputs
st.sidebar.header("Select Your Filters")
commodity = st.sidebar.selectbox("Select Commodity", sorted(df['Commodity'].unique()))

if commodity:
    states = df[df['Commodity'] == commodity]['State'].unique()
    state = st.sidebar.selectbox("Select State", sorted(states))

    if state:
        districts = df[(df['Commodity'] == commodity) & (df['State'] == state)]['District'].unique()
        district = st.sidebar.selectbox("Select District", sorted(districts))

        if district:
            st.subheader(f"Showing Details for {commodity} in {district}, {state}")

            # Filtered Data
            df_filtered = df[
                (df['Commodity'] == commodity) &
                (df['State'] == state) &
                (df['District'] == district)
            ]

            if df_filtered.empty:
                st.error("No matching data found. Please select different options.")
            else:
                # ==============================
                # Display Current Data
                # ==============================
                st.write("### 📈 Historical Price and Rainfall Data")
                st.dataframe(df_filtered[['Year', 'Market', 'Price per kg (INR)', 'Rainfall (cm)']])

                # ==============================
                # Future Year Selection
                # ==============================
                future_year = st.number_input("Select Future Year", min_value=int(df_filtered['Year'].max()) + 1, max_value=2100, value=2030)

                if st.button("Run Prediction"):
                    # ==============================
                    # Prepare Data
                    # ==============================
                    df_agg = df_filtered.groupby('Year').agg({
                        'Price per kg (INR)': 'mean',
                        'Rainfall (cm)': 'mean'
                    }).reset_index()

                    df_agg.rename(columns={'Price per kg (INR)': 'modal_price'}, inplace=True)

                    data_for_model = df_agg[['Year', 'modal_price', 'Rainfall (cm)']]

                    # Scaling
                    scaler = MinMaxScaler()
                    scaled_data = scaler.fit_transform(data_for_model)

                    # Prepare sequences
                    def create_sequences(data, look_back=3):
                        X, y = [], []
                        for i in range(len(data) - look_back):
                            X.append(data[i:i+look_back])
                            y.append(data[i+look_back, 1])  # Predict modal_price
                        return np.array(X), np.array(y)

                    look_back = 3
                    X_seq, y_seq = create_sequences(scaled_data)

                    # ==============================
                    # Build LSTM Model
                    # ==============================
                    model = Sequential([
                        Input(shape=(look_back, scaled_data.shape[1])),
                        LSTM(50, activation='relu'),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mse')
                    model.fit(X_seq, y_seq, epochs=50, verbose=0)

                    # ==============================
                    # Predict Future Prices
                    # ==============================
                    last_year = int(df_agg['Year'].max())
                    n_future = future_year - last_year

                    current_seq = scaled_data[-look_back:].copy()
                    predicted_years = []
                    predicted_prices = []

                    for _ in range(n_future):
                        pred_scaled = model.predict(current_seq[np.newaxis, :])[0]
                        # Build next year input manually
                        next_year_scaled = scaler.transform([[last_year+1, pred_scaled[0], current_seq[-1,2]]])[0]

                        predicted_years.append(last_year + 1)
                        predicted_prices.append(pred_scaled[0])

                        current_seq = np.vstack([current_seq[1:], next_year_scaled])
                        last_year += 1

                    # Inverse scale prices
                    all_data = np.vstack([scaled_data, np.column_stack([
                        np.linspace(df_agg['Year'].max()+1, future_year, len(predicted_prices)),
                        predicted_prices,
                        [scaled_data[-1,2]]*len(predicted_prices)  # same Rainfall
                    ])])

                    all_prices = scaler.inverse_transform(all_data)[:,1]

                    # ==============================
                    # Plot
                    # ==============================
                    historical_years = df_agg['Year'].tolist()
                    future_years = list(range(df_agg['Year'].max()+1, future_year+1))

                    fig, ax = plt.subplots(figsize=(12,5))
                    ax.plot(historical_years, df_agg['modal_price'], marker='o', label='Historical Prices')
                    ax.plot(future_years, all_prices[-n_future:], marker='x', linestyle='--', color='red', label='Predicted Prices')
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Modal Price (INR)')
                    ax.set_title(f"{commodity} Price Prediction for {district}, {state}")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

                    # ==============================
                    # Final Predicted Price
                    # ==============================
                    st.success(f"📌 Predicted Price of {commodity} in {district}, {state} for year {future_year} is ₹{all_prices[-1]:.2f}/kg")


