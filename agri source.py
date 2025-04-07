import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Streamlit App Title
st.title("🌾 Agricultural Modal Price Forecast")
st.markdown("Predict future modal prices for a commodity using LSTM with climate data")

# Input Section
state_input = st.text_input("Enter the State", value="Karnataka")
district_input = st.text_input("Enter the District", value="Belgaum")
commodity_input = st.text_input("Enter the Commodity", value="Tomato")
future_year = st.number_input("Enter the Future Year (e.g., 2030)", min_value=2024, max_value=2100, step=1)

# File Upload
agri_file = st.file_uploader("Upload agri.csv", type="csv")
climate_file = st.file_uploader("Upload climate.csv", type="csv")

if st.button("Run Forecast"):
    if not agri_file or not climate_file:
        st.error("Please upload both agri.csv and climate.csv files.")
        st.stop()

    # Step 1: Load and Filter agri.csv in Chunks
    agri_cols = ['state', 'district', 'commodity', 'arrival_date', 'modal_price']
    dtypes = {'state': 'category', 'district': 'category', 'commodity': 'category'}
    chunk_size = 100000
    agg_chunks = []

    for chunk in pd.read_csv(agri_file, usecols=agri_cols, dtype=dtypes, chunksize=chunk_size):
        mask = ((chunk['state'] == state_input) &
                (chunk['district'] == district_input) &
                (chunk['commodity'] == commodity_input))
        filtered = chunk.loc[mask]
        if not filtered.empty:
            agg_chunks.append(filtered)

    if len(agg_chunks) == 0:
        st.error("No matching data found in agri.csv for the specified filters.")
        st.stop()

    df_agri_filtered = pd.concat(agg_chunks, ignore_index=True)
    df_agri_filtered['arrival_date'] = pd.to_datetime(df_agri_filtered['arrival_date'], format='%d/%m/%Y', errors='coerce')
    df_agri_filtered['year'] = df_agri_filtered['arrival_date'].dt.year

    # Step 2: Aggregate Data by Year
    df_agg = df_agri_filtered.groupby('year')['modal_price'].mean().reset_index()

    # Step 3: Load and Process Climate Data
    df_climate = pd.read_csv(climate_file)
    df_climate_state = df_climate[df_climate['state'] == state_input]
    if df_climate_state.empty:
        st.error("No climate data found for state: " + state_input)
        st.stop()

    numeric_climate = df_climate_state.select_dtypes(include=[np.number])
    if numeric_climate.empty:
        st.error("No numeric climate features found for state: " + state_input)
        st.stop()

    state_climate_features = numeric_climate.iloc[0].values.astype(float)
    climate_features = numeric_climate.columns.tolist()

    for feat in climate_features:
        df_agg[feat] = float(df_climate_state.iloc[0][feat])

    cols = ['year', 'modal_price'] + climate_features
    df_agg = df_agg[cols]
    st.write("### Aggregated Data Sample")
    st.write(df_agg.head())

    # Step 4: Prepare Data for LSTM
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
        st.error(f"Not enough data to create sequences. At least {look_back + 1} records are required.")
        st.stop()

    # Step 5: Train LSTM Model
    num_features = data_scaled.shape[1]
    model = Sequential([
        Input(shape=(look_back, num_features)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_seq, y_seq, epochs=50, verbose=0)

    # Step 6: Forecast
    last_year = int(df_agg['year'].max())
    n_future = future_year - last_year
    if n_future <= 0:
        st.error("Future year must be greater than last available year: " + str(last_year))
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

    # Step 7: Visualizations
    historical_years = df_agg['year'].values
    historical_modal_prices = df_agg['modal_price'].values

    # Graph 1: Modal Price Forecast
    fig1, ax1 = plt.subplots()
    ax1.plot(historical_years, historical_modal_prices, marker='o', label='Historical')
    ax1.plot(predicted_years, predicted_modal_prices, marker='x', linestyle='--', color='red', label='Forecasted')
    ax1.annotate(f"{future_year}: {predicted_modal_prices[-1]:.2f}",
                 xy=(future_year, predicted_modal_prices[-1]),
                 xytext=(10, -15), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->"))
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Modal Price")
    ax1.legend()
    st.pyplot(fig1)

    # Graph 2: Climate Bar Chart
    fig2, ax2 = plt.subplots()
    ax2.bar(climate_features, state_climate_features, color='skyblue')
    ax2.set_ylabel("Value")
    ax2.set_title(f"Climate Features for {state_input}")
    st.pyplot(fig2)

    # Graph 3: Combined Plot
    selected_climate_feature = climate_features[0]
    combined_years = np.concatenate([historical_years, predicted_years])
    selected_value = state_climate_features[climate_features.index(selected_climate_feature)]
    climate_line = np.full(combined_years.shape, selected_value)

    fig3, ax3 = plt.subplots()
    ax3.plot(historical_years, historical_modal_prices, label='Historical Price', color='tab:blue')
    ax3.plot(predicted_years, predicted_modal_prices, linestyle='--', label='Forecasted Price', color='tab:red')
    ax4 = ax3.twinx()
    ax4.plot(combined_years, climate_line, linestyle=':', color='tab:green', label=selected_climate_feature)
    fig3.tight_layout()
    st.pyplot(fig3)

    # Graph 4: Correlation Heatmap
    fig4, ax4 = plt.subplots()
    corr = df_agg.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax4)
    ax4.set_title("Correlation Heatmap")
    st.pyplot(fig4)

    # Final Output
    st.success(f"Predicted Modal Price in {future_year}: ₹{predicted_modal_prices[-1]:.2f}")
