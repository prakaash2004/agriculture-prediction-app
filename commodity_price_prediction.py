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

# Load dataset
df = pd.read_csv('agrio.csv')
df_2025 = df[df['Year'] == 2025]

st.title("📊 Agri Commodity Price Explorer & LSTM Prediction")

# User Inputs
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
            # Show Top 3 Markets for Commodity
            st.subheader("🔝 Top 3 Markets by Price (2025)")
            top3 = df_2025[df_2025['Commodity'] == commodity].sort_values('Price per kg (INR)', ascending=False).head(3)
            st.dataframe(top3[['State', 'District', 'Market', 'Price per kg (INR)']])

            # Show Top 5 in selected State
            st.subheader("🏆 Top 5 Prices in Selected State")
            top5 = df_2025[
                (df_2025['Commodity'] == commodity) & 
                (df_2025['State'] == state)
            ].sort_values('Price per kg (INR)', ascending=False).head(5)
            st.dataframe(top5[['District', 'Market', 'Price per kg (INR)']])

            # All Markets in Selected District
            st.subheader("📍 Markets in Selected District")
            market_df = df_2025[
                (df_2025['Commodity'] == commodity) &
                (df_2025['State'] == state) &
                (df_2025['District'] == district)
            ].sort_values('Price per kg (INR)', ascending=False)
            st.dataframe(market_df[['Market', 'Price per kg (INR)']])

            # Future Price Prediction
            st.subheader("📈 Predict Future Price")
            future_year = st.number_input("Select Future Year", min_value=2025, max_value=2100, value=2030, step=1)
            if st.button("Run LSTM Prediction"):
                df_filtered = df[
                    (df['State'] == state) &
                    (df['District'] == district) &
                    (df['Commodity'] == commodity)
                ]

                if df_filtered.empty:
                    st.error("No matching data found.")
                else:
                    df_agg = df_filtered.groupby('Year')['Price per kg (INR)'].mean().reset_index()
                    df_agg.rename(columns={'Price per kg (INR)': 'modal_price'}, inplace=True)

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

                    model = Sequential([
                        Input(shape=(look_back, scaled_data.shape[1])),
                        LSTM(50, activation='relu'),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mse')
                    model.fit(X_seq, y_seq, epochs=50, verbose=0)

                    last_year = df_agg['Year'].max()
                    n_future = future_year - last_year

                    if n_future <= 0:
                        st.warning(f"Future year must be greater than {last_year}.")
                    else:
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

                        # Smoothing
                        smoothed_prices = [predicted_prices[0]]
                        for i in range(1, len(predicted_prices)):
                            prev_price = smoothed_prices[-1]
                            curr_price = predicted_prices[i]
                            if curr_price < prev_price * 0.97:
                                curr_price = prev_price * 0.97
                            smoothed_prices.append(curr_price)
                        predicted_prices = smoothed_prices

                        # Line Plot
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(df_agg['Year'], df_agg['modal_price'], marker='o', label='Historical')
                        ax.plot(predicted_years, predicted_prices, marker='x', linestyle='--', color='red', label='Predicted')
                        ax.set_title(f"{commodity} Price Forecast in {district}, {state}")
                        ax.set_xlabel("Year")
                        ax.set_ylabel("Price (INR)")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)

                        # Heatmap
                        all_years = list(df_agg['Year']) + predicted_years
                        all_prices = list(df_agg['modal_price']) + predicted_prices
                        heat_df = pd.DataFrame({'Year': all_years, 'Price': all_prices})
                        heat_df['Category'] = ['Historical'] * len(df_agg) + ['Predicted'] * len(predicted_years)
                        pivot = heat_df.pivot_table(index='Category', columns='Year', values='Price')

                        fig2, ax2 = plt.subplots(figsize=(12, 2))
                        sns.heatmap(pivot, annot=True, fmt=".1f", cmap='coolwarm', ax=ax2)
                        st.pyplot(fig2)

                        st.success(f"📌 Predicted modal price of {commodity} in {district}, {state} for the year {future_year} is ₹{predicted_prices[-1]:.2f}/kg")
