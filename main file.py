import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------
# Step 1: User Inputs
# ---------------------------------------------
st.title("🌾 Agriculture Modal Price Forecasting (Simple Version)")
state_input = st.text_input("Enter the State:")
district_input = st.text_input("Enter the District:")
commodity_input = st.text_input("Enter the Commodity:")
future_year = st.number_input("Enter the Future Year (e.g., 2035):", min_value=2024, step=1)

# Load data only if inputs are provided
if state_input and district_input and commodity_input and future_year:
    # ---------------------------------------------
    # Step 2: Load and filter agri.csv
    # ---------------------------------------------
    try:
        df = pd.read_csv("agri.csv", usecols=['state', 'district', 'commodity', 'arrival_date', 'modal_price'])
    except:
        st.error("🚫 Error loading agri.csv")
        st.stop()

    mask = (
        (df['state'] == state_input) &
        (df['district'] == district_input) &
        (df['commodity'] == commodity_input)
    )

    df_filtered = df[mask].copy()
    df_filtered['arrival_date'] = pd.to_datetime(df_filtered['arrival_date'], errors='coerce', dayfirst=True)
    df_filtered = df_filtered.dropna(subset=['arrival_date', 'modal_price'])
    df_filtered['year'] = df_filtered['arrival_date'].dt.year
    df_yearly = df_filtered.groupby('year')['modal_price'].mean().reset_index()

    if df_yearly.empty:
        st.warning("⚠️ No data found for the given filters.")
        st.stop()

    # ---------------------------------------------
    # Step 3: Predict using simple linear regression (manual)
    # ---------------------------------------------
    years = df_yearly['year'].values
    prices = df_yearly['modal_price'].values

    # Manual linear regression using numpy
    coeffs = np.polyfit(years, prices, deg=1)
    trend = np.poly1d(coeffs)

    # Predict future years
    last_year = years.max()
    prediction_years = np.arange(last_year + 1, future_year + 1)
    predicted_prices = trend(prediction_years)

    # Combine data
    all_years = np.concatenate([years, prediction_years])
    all_prices = np.concatenate([prices, predicted_prices])
    label = ['Historical'] * len(years) + ['Predicted'] * len(prediction_years)

    df_combined = pd.DataFrame({
        'Year': all_years,
        'Modal Price': all_prices,
        'Type': label
    })

    # ---------------------------------------------
    # Step 4: Display outputs
    # ---------------------------------------------
    st.subheader("📊 Historical and Predicted Prices")
    historical = df_combined[df_combined['Type'] == 'Historical'].set_index('Year')
    predicted = df_combined[df_combined['Type'] == 'Predicted'].set_index('Year')

    st.line_chart(historical['Modal Price'])
    st.line_chart(predicted['Modal Price'])

    st.metric(f"📌 Predicted Modal Price for {future_year}", f"{predicted_prices[-1]:.2f}")

    # ---------------------------------------------
    # Optional: Climate integration
    # ---------------------------------------------
    try:
        df_climate = pd.read_csv("climate.csv")
        df_state_climate = df_climate[df_climate['state'] == state_input]
        numeric_features = df_state_climate.select_dtypes(include=np.number).iloc[0]

        st.subheader("🌦️ Climate Features")
        st.dataframe(numeric_features.to_frame("Value"))
        st.bar_chart(numeric_features)
    except:
        st.info("ℹ️ Climate data not available or could not be loaded.")

    # ---------------------------------------------
    # Raw data preview
    # ---------------------------------------------
    st.subheader("🔍 Raw Historical Data")
    st.dataframe(df_yearly)
