import streamlit as st
import pandas as pd
import numpy as np

# Combine historical and predicted data
combined_years = np.concatenate([historical_years, predicted_years])
combined_prices = np.concatenate([historical_modal_prices, predicted_modal_prices])
price_type = ['Historical'] * len(historical_years) + ['Predicted'] * len(predicted_years)

df_price = pd.DataFrame({
    'Year': combined_years,
    'Modal Price': combined_prices,
    'Type': price_type
})

# 1️⃣ Modal Price Line Chart
st.subheader(f"📈 Modal Price Prediction - {commodity_input} in {district_input}, {state_input}")
historical_df = df_price[df_price['Type'] == 'Historical'].set_index('Year')
predicted_df = df_price[df_price['Type'] == 'Predicted'].set_index('Year')

st.line_chart(historical_df[['Modal Price']], use_container_width=True)
st.line_chart(predicted_df[['Modal Price']], use_container_width=True)

# Predicted value for the future year
st.metric(label=f"📌 Predicted Modal Price in {future_year}", value=f"{predicted_modal_prices[-1]:.2f}")

# 2️⃣ Climate Features Bar Chart
st.subheader("🌦️ Climate Features")
df_climate_bar = pd.DataFrame({
    'Climate Feature': climate_features,
    'Value': state_climate_features
}).set_index('Climate Feature')

st.bar_chart(df_climate_bar)

# 3️⃣ Combined Table (Price and One Climate Feature)
st.subheader(f"📊 Table: Price vs {selected_climate_feature}")
df_combined = pd.DataFrame({
    'Year': combined_years,
    'Modal Price': combined_prices,
    selected_climate_feature: climate_line
})
st.dataframe(df_combined)

# 4️⃣ Correlation Table (instead of heatmap)
st.subheader("📘 Correlation Table")
correlation = df_agg.corr().round(2)
st.dataframe(correlation)
