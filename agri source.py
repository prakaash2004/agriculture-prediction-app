import pandas as pd
import streamlit as st

# -------------------------------
# Step 1: Load and verify the dataset structure
# -------------------------------
def load_data():
    try:
        data = pd.read_csv('agri.csv', delimiter=',', header=0)

        if len(data.columns) == 1:
            data = data[data.columns[0]].str.split(',', expand=True)
            data.columns = ['state', 'district', 'market', 'commodity', 'arrival_date',
                            'min_price', 'max_price', 'modal_price', 'extra_col_1', 'extra_col_2']

        # Convert price columns to numeric
        for col in ['min_price', 'max_price', 'modal_price']:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        return data

    except FileNotFoundError:
        st.error("Error: The file 'agri.csv' was not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()


data = load_data()

# Extract unique dropdown options
unique_states = sorted(data['state'].dropna().unique())
unique_districts = sorted(data['district'].dropna().unique())
unique_commodities = sorted(data['commodity'].dropna().unique())

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Agriculture Market Price Explorer")

commodity_input = st.selectbox("Select Commodity:", unique_commodities)
state_input = st.selectbox("Select State:", unique_states)
district_input = st.selectbox("Select District:", unique_districts)

# -------------------------------
# Step 1: Top 3 Prices in India for Selected Commodity
# -------------------------------
st.subheader(f"Top 3 Modal Prices in India for '{commodity_input}'")
top_commodity_data = data[data['commodity'] == commodity_input].dropna(subset=['modal_price'])
top_prices = top_commodity_data.nlargest(3, 'modal_price')

if top_prices.empty:
    st.warning("No data found for this commodity.")
else:
    st.dataframe(top_prices[['state', 'market', 'modal_price']])

# -------------------------------
# Step 2: Top Market per District in Selected State
# -------------------------------
st.subheader(f"Top Market per District in '{state_input}' for '{commodity_input}'")
state_data = data[(data['state'] == state_input) & (data['commodity'] == commodity_input)]
state_data = state_data.dropna(subset=['modal_price'])

if state_data.empty:
    st.warning("No data found for this state and commodity.")
else:
    top_markets = state_data.loc[state_data.groupby('district')['modal_price'].idxmax()]
    top_markets = top_markets.sort_values('modal_price', ascending=False)
    st.dataframe(top_markets[['district', 'market', 'modal_price']])

# -------------------------------
# Step 3: All Markets in Selected District
# -------------------------------
st.subheader(f"All Markets in '{district_input}' for '{commodity_input}'")
district_data = data[(data['district'] == district_input) & (data['commodity'] == commodity_input)]
district_data = district_data.dropna(subset=['modal_price'])
district_data = district_data.sort_values('modal_price', ascending=False)

if district_data.empty:
    st.warning("No data found for this district and commodity.")
else:
    st.dataframe(district_data[['market', 'modal_price']])
