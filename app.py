import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('shill_bidding_model.h5')

# Load the scaler (you need to save this during training)
scaler = StandardScaler()
# Load scaler parameters (you need to save these during training)
scaler.mean_ = np.load('scaler_mean.npy')
scaler.scale_ = np.load('scaler_scale.npy')

st.title('Shill Bidding Prediction')

# Create input fields for each feature
st.header('Enter Feature Values:')

bid_ratio = st.number_input('Bid Ratio', min_value=0.0, max_value=1.0, step=0.01)
bidding_tendency = st.number_input('Bidding Tendency', min_value=0.0, max_value=1.0, step=0.01)
successive_outbidding = st.number_input('Successive Outbidding', min_value=0.0, max_value=1.0, step=0.01)
last_bidding = st.number_input('Last Bidding', min_value=0.0, max_value=1.0, step=0.01)
auction_bids = st.number_input('Auction Bids', min_value=0, step=1)
starting_price_average = st.number_input('Starting Price Average', min_value=0.0, step=0.01)
early_bidding = st.number_input('Early Bidding', min_value=0.0, max_value=1.0, step=0.01)
winning_ratio = st.number_input('Winning Ratio', min_value=0.0, max_value=1.0, step=0.01)
auction_duration = st.number_input('Auction Duration', min_value=0, step=1)

if st.button('Predict'):
    # Prepare input data
    input_data = np.array([[
        bid_ratio, bidding_tendency, successive_outbidding, last_bidding,
        auction_bids, starting_price_average, early_bidding, winning_ratio,
        auction_duration
    ]])

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display result
    st.header('Prediction Result:')
    if prediction[0][0] > 0.5:
        st.write('This bidding behavior is predicted to be: **Shill Bidding**')
        st.write(f'Confidence: {prediction[0][0]:.2f}')
    else:
        st.write('This bidding behavior is predicted to be: **Normal Bidding**')
        st.write(f'Confidence: {1 - prediction[0][0]:.2f}')