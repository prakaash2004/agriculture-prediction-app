import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Agricultural Modal Price Forecasting (No ML Libraries)")

# -------------------------------
# 1. User Inputs
# -------------------------------
state_input = st.text_input("Enter the State:")
district_input = st.text_input("Enter the District:")
commodity_input = st.text_input("Enter the Commodity:")
future_year = st.number_input("Enter the Future Year (e.g., 2100):", step=1)

if state_input and district_input and commodity_input and future_year:
    # -------------------------------
    # 2. Load and Filter agri.csv
    # -------------------------------
    try:
        agri_cols = ['state', 'district', 'commodity', 'arrival_date', 'modal_price']
        data = pd.read_csv("agri.csv", usecols=agri_cols)
    except Exception as e:
        st.error(f"Error reading agri.csv: {e}")
        st.stop()

    mask = (
        (data['state'] == state_input) &
        (data['district'] == district_input) &
        (data['commodity'] == commodity_input)
    )
    filtered = data[mask]

    if filtered.empty:
        st.warning("No matching data found.")
        st.stop()

    filtered['arrival_date'] = pd.to_datetime(filtered['arrival_date'], format='%d/%m/%Y', errors='coerce')
    filtered['year'] = filtered['arrival_date'].dt.year
    df_agg = filtered.groupby('year')['modal_price'].mean().reset_index()

    # -------------------------------
    # 3. Load Climate Data
    # -------------------------------
    try:
        df_climate = pd.read_csv("climate.csv")
    except Exception as e:
        st.error(f"Error reading climate.csv: {e}")
        st.stop()

    df_clim_state = df_climate[df_climate['state'] == state_input]
    if df_clim_state.empty:
        st.warning("No climate data for selected state.")
        st.stop()

    numeric_features = df_clim_state.select_dtypes(include=[np.number]).iloc[0]
    climate_features = numeric_features.index.tolist()

    for feat in climate_features:
        df_agg[feat] = numeric_features[feat]

    # -------------------------------
    # 4. Forecast using NumPy Linear Regression
    # -------------------------------
    years = df_agg['year'].values
    prices = df_agg['modal_price'].values

    if len(years) < 2:
        st.warning("Not enough data points for prediction.")
        st.stop()

    # Linear regression via NumPy: y = mx + c
    X = np.vstack([years, np.ones_like(years)]).T
    m, c = np.linalg.lstsq(X, prices, rcond=None)[0]

    future_years = np.arange(years[-1] + 1, int(future_year) + 1)
    future_prices = m * future_years + c

    st.subheader("Forecasted Price for Year {}: ₹{:.2f}".format(future_year, future_prices[-1]))

    # -------------------------------
    # 5. Plotting
    # -------------------------------
    st.subheader("1. Modal Price Forecast")

    fig1, ax1 = plt.subplots()
    ax1.plot(years, prices, marker='o', label='Historical')
    ax1.plot(future_years, future_prices, marker='x', linestyle='--', color='red', label='Forecasted')
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Modal Price")
    ax1.set_title(f"Modal Price Forecast for {commodity_input}")
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("2. Climate Features (State-level)")
    fig2, ax2 = plt.subplots()
    ax2.bar(climate_features, numeric_features.values)
    ax2.set_xticklabels(climate_features, rotation=45, ha='right')
    ax2.set_ylabel("Value")
    ax2.set_title(f"Climate Features for {state_input}")
    st.pyplot(fig2)

    st.subheader("3. Climate vs Modal Price (First Climate Feature)")

    selected_feature = climate_features[0]
    fig3, ax3 = plt.subplots()
    ax3.plot(df_agg['year'], df_agg['modal_price'], label='Modal Price', color='blue')
    ax3.set_ylabel("Modal Price", color='blue')
    ax4 = ax3.twinx()
    ax4.plot(df_agg['year'], df_agg[selected_feature], label=selected_feature, color='green', linestyle='--')
    ax4.set_ylabel(selected_feature, color='green')
    fig3.suptitle("Price vs " + selected_feature)
    st.pyplot(fig3)

    st.subheader("4. Correlation Heatmap")
    fig4, ax5 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_agg.corr(), annot=True, cmap='coolwarm', ax=ax5)
    ax5.set_title("Correlation between Modal Price and Climate Features")
    st.pyplot(fig4)
