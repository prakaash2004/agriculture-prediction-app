import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -------------------------------
# Title & Inputs
# -------------------------------
st.title("🌾 Agricultural Modal Price Forecasting")

state_input = st.text_input("Enter the State")
district_input = st.text_input("Enter the District")
commodity_input = st.text_input("Enter the Commodity")
future_year = st.number_input("Enter the future year (e.g., 2030):", min_value=2025, max_value=2100, step=1)

# -------------------------------
# Load agri.csv
# -------------------------------
@st.cache_data
def load_agri_data():
    try:
        agri_cols = ['state', 'district', 'commodity', 'arrival_date', 'modal_price']
        df = pd.read_csv("agri.csv", usecols=agri_cols)
        df['arrival_date'] = pd.to_datetime(df['arrival_date'], format="%d/%m/%Y", errors='coerce')
        df = df.dropna(subset=['arrival_date', 'modal_price'])
        df['year'] = df['arrival_date'].dt.year
        return df
    except Exception as e:
        return None

agri_df = load_agri_data()

# -------------------------------
# Load climate.csv
# -------------------------------
@st.cache_data
def load_climate_data():
    try:
        return pd.read_csv("climate.csv")
    except Exception as e:
        return None

climate_df = load_climate_data()

# -------------------------------
# Process & Forecast
# -------------------------------
if agri_df is not None and climate_df is not None and state_input and district_input and commodity_input:
    filtered = agri_df[
        (agri_df['state'] == state_input) &
        (agri_df['district'] == district_input) &
        (agri_df['commodity'] == commodity_input)
    ]

    if filtered.empty:
        st.warning("No matching records found in agri.csv.")
    else:
        # Group by year and get average modal price
        yearly_data = filtered.groupby('year')['modal_price'].mean().reset_index()

        # Climate data for that state
        climate_state = climate_df[climate_df['state'] == state_input]
        climate_features = climate_state.select_dtypes(include=[np.number])

        if climate_state.empty or climate_features.empty:
            st.warning("No climate data found for this state.")
        else:
            # Use linear regression (NumPy) for simple forecast
            X = yearly_data['year'].values
            y = yearly_data['modal_price'].values
            A = np.vstack([X, np.ones_like(X)]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]

            # Predict from last year up to future year
            last_year = X.max()
            predict_years = np.arange(last_year + 1, future_year + 1)
            predict_prices = m * predict_years + c

            # Merge historical + predicted
            all_years = np.concatenate([X, predict_years])
            all_prices = np.concatenate([y, predict_prices])

            # -------------------------------
            # Plot Modal Price Forecast
            # -------------------------------
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=X, y=y, mode='lines+markers', name='Historical'))
            fig1.add_trace(go.Scatter(x=predict_years, y=predict_prices, mode='lines+markers', name='Forecast', line=dict(dash='dash')))
            fig1.update_layout(title='📈 Modal Price Forecast',
                               xaxis_title='Year', yaxis_title='Modal Price')

            # -------------------------------
            # Climate Bar Chart
            # -------------------------------
            fig2 = go.Figure()
            for col in climate_features.columns:
                fig2.add_trace(go.Bar(x=[col], y=[climate_features[col].values[0]]))
            fig2.update_layout(title=f"🌦️ Climate Features for {state_input}", yaxis_title='Value')

            # -------------------------------
            # Combined Trend
            # -------------------------------
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=all_years, y=all_prices, mode='lines+markers', name='Price'))
            first_climate = climate_features.columns[0]
            climate_val = climate_features[first_climate].values[0]
            fig3.add_trace(go.Scatter(x=all_years, y=[climate_val]*len(all_years), mode='lines', name=first_climate, line=dict(dash='dot')))
            fig3.update_layout(title='🔗 Combined Price & Climate Feature Trend',
                               xaxis_title='Year')

            # -------------------------------
            # Show Everything
            # -------------------------------
            st.subheader("📊 Forecast Results")
            st.plotly_chart(fig1)
            st.subheader("🌡️ Climate Overview")
            st.plotly_chart(fig2)
            st.subheader("🧬 Combined Trend")
            st.plotly_chart(fig3)

            st.success(f"✅ Predicted Modal Price for {future_year}: ₹{predict_prices[-1]:.2f}")
