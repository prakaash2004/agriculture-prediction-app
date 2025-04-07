import streamlit as st
import pandas as pd
import numpy as np
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# Streamlit UI
# =============================================================================
st.title("Agriculture Price Forecasting App (No Visuals)")

state_input = st.text_input("Enter State", value="Karnataka")
district_input = st.text_input("Enter District", value="Belgaum")
commodity_input = st.text_input("Enter Commodity", value="Tomato")
future_year = st.number_input("Enter Future Year", min_value=2025, max_value=2100, value=2030)

if st.button("Run Forecast"):
    try:
        agri_cols = ['state', 'district', 'commodity', 'arrival_date', 'modal_price']
        dtypes = {'state': 'category', 'district': 'category', 'commodity': 'category'}
        chunk_size = 100000
        agg_chunks = []

        for chunk in pd.read_csv("agri.csv", usecols=agri_cols, dtype=dtypes, chunksize=chunk_size):
            mask = (
                (chunk['state'] == state_input) &
                (chunk['district'] == district_input) &
                (chunk['commodity'] == commodity_input)
            )
            filtered = chunk.loc[mask]
            if not filtered.empty:
                agg_chunks.append(filtered)

        if not agg_chunks:
            st.error("No matching data found in agri.csv for the specified filters.")
            st.stop()

        df_agri_filtered = pd.concat(agg_chunks, ignore_index=True)
        df_agri_filtered['arrival_date'] = pd.to_datetime(
            df_agri_filtered['arrival_date'], format='%d/%m/%Y', errors='coerce')
        df_agri_filtered['year'] = df_agri_filtered['arrival_date'].dt.year
        df_agg = df_agri_filtered.groupby('year')['modal_price'].mean().reset_index()

        # Climate data
        df_climate = pd.read_csv("climate.csv")
        df_climate_state = df_climate[df_climate['state'] == state_input]

        if df_climate_state.empty:
            st.error("No climate data found for state.")
            st.stop()

        numeric_climate = df_climate_state.select_dtypes(include=[np.number])
        if numeric_climate.empty:
            st.error("No numeric climate features found.")
            st.stop()

        climate_features = numeric_climate.columns.tolist()
        state_climate_features = numeric_climate.iloc[0].values.astype(float)

        for feat in climate_features:
            df_agg[feat] = float(df_climate_state.iloc[0][feat])

        df_agg = df_agg[['year', 'modal_price'] + climate_features]

        # =================== LSTM Preparation ====================
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
            st.error(f"Not enough data for look_back={look_back}")
            st.stop()

        num_features = data_scaled.shape[1]
        model = Sequential([
            Input(shape=(look_back, num_features)),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_seq, y_seq, epochs=50, verbose=0)

        # =================== Forecast ====================
        last_year = int(df_agg['year'].max())
        n_future = future_year - last_year

        if n_future <= 0:
            st.error(f"Future year must be greater than the last year in data ({last_year})")
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

        # =================== Output ====================
        forecast_df = pd.DataFrame({
            'Year': predicted_years,
            'Predicted Modal Price': np.round(predicted_modal_prices, 2)
        })

        st.success(f"Forecast completed up to {future_year}.")
        st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
