import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Agricultural Price Forecasting (No TensorFlow)")
st.markdown("Forecast modal prices using historical and climate data with simple regression.")

# =======================
# 1. User Inputs
# =======================
state_input = st.text_input("Enter the State:")
district_input = st.text_input("Enter the District:")
commodity_input = st.text_input("Enter the Commodity:")
future_year = st.number_input("Enter the Future Year to Predict:", min_value=2025, step=1)

# =======================
# 2. Load Data
# =======================
@st.cache_data
def load_agri_data():
    agri_cols = ['state', 'district', 'commodity', 'arrival_date', 'modal_price']
    dtypes = {'state': 'category', 'district': 'category', 'commodity': 'category'}
    chunks = []
    for chunk in pd.read_csv("agri.csv", usecols=agri_cols, dtype=dtypes, chunksize=100000):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

@st.cache_data
def load_climate_data():
    return pd.read_csv("climate.csv")

if state_input and district_input and commodity_input:
    data = load_agri_data()
    data['arrival_date'] = pd.to_datetime(data['arrival_date'], errors='coerce', format='%d/%m/%Y')
    data['year'] = data['arrival_date'].dt.year
    filtered = data[
        (data['state'] == state_input) &
        (data['district'] == district_input) &
        (data['commodity'] == commodity_input)
    ].dropna(subset=['modal_price', 'year'])

    if filtered.empty:
        st.error("No matching data found.")
        st.stop()

    df_agg = filtered.groupby('year')['modal_price'].mean().reset_index()

    df_climate = load_climate_data()
    climate_state = df_climate[df_climate['state'] == state_input]

    if climate_state.empty:
        st.error("No climate data found for selected state.")
        st.stop()

    numeric_climate = climate_state.select_dtypes(include=[np.number])
    if numeric_climate.empty:
        st.error("No numeric climate features found.")
        st.stop()

    state_climate_features = numeric_climate.iloc[0].to_dict()
    for k, v in state_climate_features.items():
        df_agg[k] = v

    # =======================
    # 3. Linear Regression
    # =======================
    features = ['year'] + list(numeric_climate.columns)
    X = df_agg[features]
    y = df_agg['modal_price']

    model = LinearRegression()
    model.fit(X, y)

    # Forecast
    last_year = df_agg['year'].max()
    predict_years = np.arange(last_year + 1, future_year + 1)
    future_df = pd.DataFrame({'year': predict_years})
    for feat in numeric_climate.columns:
        future_df[feat] = state_climate_features[feat]

    future_preds = model.predict(future_df)

    # =======================
    # 4. Plotting
    # =======================
    st.subheader("Modal Price Forecast")
    fig1, ax1 = plt.subplots()
    ax1.plot(df_agg['year'], df_agg['modal_price'], marker='o', label='Historical')
    ax1.plot(predict_years, future_preds, marker='x', linestyle='--', color='red', label='Predicted')
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Modal Price")
    ax1.set_title(f"Forecast for {commodity_input} in {district_input}, {state_input}")
    ax1.legend()
    st.pyplot(fig1)

    # Climate bar chart
    st.subheader("Climate Features Used")
    fig2, ax2 = plt.subplots()
    ax2.bar(state_climate_features.keys(), state_climate_features.values(), color='skyblue')
    ax2.set_ylabel("Value")
    ax2.set_title(f"Climate Features for {state_input}")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df_agg[['modal_price'] + list(numeric_climate.columns)].corr()
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax3)
    st.pyplot(fig3)

    # Final predicted value
    st.success(f"Predicted modal price in {future_year}: ₹{future_preds[-1]:.2f}")
