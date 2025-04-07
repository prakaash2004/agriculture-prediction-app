import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1️⃣ Line Chart: Historical + Predicted Modal Prices
st.subheader("📈 Modal Price Prediction")

combined_years = np.concatenate([historical_years, predicted_years])
combined_prices = np.concatenate([historical_modal_prices, predicted_modal_prices])
price_type = ['Historical'] * len(historical_years) + ['Predicted'] * len(predicted_years)

df_price_plot = pd.DataFrame({
    'Year': combined_years,
    'Modal Price': combined_prices,
    'Type': price_type
})

fig_price = px.line(df_price_plot, x='Year', y='Modal Price', color='Type', markers=True,
                    title=f'Modal Price Forecast for {commodity_input} in {district_input}, {state_input}')
st.plotly_chart(fig_price, use_container_width=True)

# Annotate the last predicted year with a metric
st.metric(label=f"💰 Predicted Modal Price in {future_year}", value=f"{predicted_modal_prices[-1]:.2f}")

# 2️⃣ Bar Chart: Climate Features
st.subheader("🌦️ Climate Features")
df_climate_bar = pd.DataFrame({
    'Feature': climate_features,
    'Value': state_climate_features
})

st.bar_chart(df_climate_bar.set_index('Feature'))

# 3️⃣ Dual-Axis Simulation: Price and One Climate Feature
st.subheader(f"📊 Modal Price vs Climate Feature: {selected_climate_feature}")
df_combined = pd.DataFrame({
    'Year': combined_years,
    'Modal Price': combined_prices,
    selected_climate_feature: climate_line
})

fig_dual = go.Figure()
fig_dual.add_trace(go.Scatter(x=df_combined['Year'], y=df_combined['Modal Price'],
                              mode='lines+markers', name='Modal Price', yaxis='y1'))
fig_dual.add_trace(go.Scatter(x=df_combined['Year'], y=df_combined[selected_climate_feature],
                              mode='lines', name=selected_climate_feature, yaxis='y2'))

fig_dual.update_layout(
    title='Price vs Climate Feature',
    xaxis=dict(title='Year'),
    yaxis=dict(title='Modal Price', side='left'),
    yaxis2=dict(title=selected_climate_feature, overlaying='y', side='right'),
    legend=dict(x=0.01, y=0.99)
)
st.plotly_chart(fig_dual, use_container_width=True)

# 4️⃣ Correlation Table (as an alternative to heatmap)
st.subheader("📘 Correlation Table")
correlation = df_agg.corr().round(2)
st.dataframe(correlation)
