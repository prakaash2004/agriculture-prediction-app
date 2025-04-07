import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# =========================== Streamlit UI ===========================
st.title("🌾 Modal Price Forecasting with Climate Data")

uploaded_agri = st.file_uploader("Upload agri.csv", type="csv")
uploaded_climate = st.file_uploader("Upload climate.csv", type="csv")

state_input = st.text_input("Enter the State")
district_input = st.text_input("Enter the District")
commodity_input = st.text_input("Enter the Commodity")
future_year = st.number_input("Enter Future Year", min_value=2025, step=1)

if uploaded_agri and uploaded_climate and state_input and district_input and commodity_input and future_year:

    # =========================== Read Agri Data ===========================
    agri_cols = ['state', 'district', 'commodity', 'arrival_date', 'modal_price']
    dtypes = {'state': 'category', 'district': 'category', 'commodity': 'category'}

    chunks = []
    chunk_size = 100000

    for chunk in pd.read_csv(uploaded_agri, usecols=agri_cols, dtype=dtypes, chunksize=chunk_size):
        mask = ((chunk['state'] == state_input) &
                (chunk['district'] == district_input) &
                (chunk['commodity'] == commodity_input))
        filtered = chunk.loc[mask]
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        st.error("No matching data found for given filters.")
        st.stop()

    df_agri_filtered = pd.concat(chunks, ignore_index=True)
    df_agri_filtered['arrival_date'] = pd.to_datetime(df_agri_filtered['arrival_date'],
                                                       format='%d/%m/%Y', errors='coerce')
    df_agri_filtered['year'] = df_agri_filtered['arrival_date'].dt.year
    df_agg = df_agri_filtered.groupby('year')['modal_price'].mean().reset_index()

    # =========================== Read Climate Data ===========================
    df_climate = pd.read_csv(uploaded_climate)
    df_climate_state = df_climate[df_climate['state'] == state_input]

    if df_climate_state.empty:
        st.error("No climate data found for this state.")
        st.stop()

    numeric_climate = df_climate_state.select_dtypes(include=[np.number])
    state_climate_features = numeric_climate.iloc[0].values.astype(float)
    climate_features = numeric_climate.columns.tolist()

    for feat in climate_features:
        df_agg[feat] = float(df_climate_state.iloc[0][feat])

    # =========================== Data Prep ===========================
    df_agg = df_agg[['year', 'modal_price'] + climate_features]
    data_raw = df_agg.values

    scaler_X = MinMaxScaler()
    data_scaled = scaler_X.fit_transform(data_raw)

    scaler_y = MinMaxScaler()
    y_raw = df_agg['modal_price'].values.reshape(-1, 1)
    y_scaled = scaler_y.fit_transform(y_raw)

    look_back = 3

    def create_sequences(data, target, look_back):
        X_seq, y_seq = [], []
        for i in range(len(data) - look_back):
            X_seq.append(data[i:i+look_back])
            y_seq.append(target[i+look_back])
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences(data_scaled, y_scaled, look_back)

    if len(X_seq) == 0:
        st.error(f"Not enough data. Need at least {look_back + 1} records.")
        st.stop()

    # =========================== LSTM Model ===========================
    num_features = data_scaled.shape[1]

    model = Sequential([
        Input(shape=(look_back, num_features)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_seq, y_seq, epochs=50, verbose=0)

    # =========================== Forecasting ===========================
    last_year = int(df_agg['year'].max())
    n_future = int(future_year - last_year)

    if n_future <= 0:
        st.error("Future year must be greater than last historical year.")
        st.stop()

    current_seq = data_scaled[-look_back:].copy()
    predicted_years = []
    predicted_modal_prices = []

    for i in range(n_future):
        pred_scaled = model.predict(current_seq[np.newaxis, :], verbose=0)
        pred_modal_raw = scaler_y.inverse_transform(pred_scaled)[0, 0]

        new_year = last_year + i + 1
        predicted_years.append(new_year)
        predicted_modal_prices.append(pred_modal_raw)

        new_row_raw = np.concatenate(([new_year, pred_modal_raw], state_climate_features))
        new_row_scaled = scaler_X.transform(new_row_raw.reshape(1, -1))[0]
        current_seq = np.vstack([current_seq[1:], new_row_scaled])

    # =========================== Graph 1: Price Prediction ===========================
    historical_years = df_agg['year'].values
    historical_modal_prices = df_agg['modal_price'].values

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(historical_years, historical_modal_prices, marker='o', label='Historical')
    ax1.plot(predicted_years, predicted_modal_prices, marker='x', linestyle='--', color='red', label='Predicted')
    ax1.annotate(f"{future_year}: {predicted_modal_prices[-1]:.2f}",
                 xy=(future_year, predicted_modal_prices[-1]),
                 xytext=(10, -15), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", color='black'))
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Modal Price')
    ax1.set_title(f'Modal Price Forecast: {commodity_input} in {district_input}, {state_input}')
    ax1.legend()
    st.pyplot(fig1)

    # =========================== Graph 2: Climate Features ===========================
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(climate_features, state_climate_features, color='skyblue')
    ax2.set_xlabel("Climate Feature")
    ax2.set_ylabel("Value")
    ax2.set_title(f"Climate Data for {state_input}")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # =========================== Graph 3: Combined Modal + Climate ===========================
    combined_years = np.concatenate([historical_years, predicted_years])
    selected_feat = climate_features[0]
    climate_line = np.full(len(combined_years), state_climate_features[climate_features.index(selected_feat)])

    fig3, ax3a = plt.subplots(figsize=(12, 5))
    ax3a.plot(historical_years, historical_modal_prices, marker='o', color='tab:blue', label='Historical')
    ax3a.plot(predicted_years, predicted_modal_prices, marker='x', linestyle='--', color='tab:red', label='Predicted')
    ax3a.set_ylabel('Modal Price', color='tab:blue')
    ax3a.set_xlabel('Year')
    ax3a.tick_params(axis='y', labelcolor='tab:blue')

    ax3b = ax3a.twinx()
    ax3b.plot(combined_years, climate_line, linestyle=':', color='tab:green', label=selected_feat)
    ax3b.set_ylabel(selected_feat, color='tab:green')
    ax3b.tick_params(axis='y', labelcolor='tab:green')
    fig3.suptitle(f'Modal Price vs {selected_feat}')
    st.pyplot(fig3)

    # =========================== Graph 4: Correlation Heatmap ===========================
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    correlation = df_agg.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", ax=ax4)
    ax4.set_title('Correlation Heatmap')
    st.pyplot(fig4)
