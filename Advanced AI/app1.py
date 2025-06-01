import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import requests
from streamlit_lottie import st_lottie

# Title and description
st.title("Automated Fraud Detection System Web App")
st.write("""
This app helps identify potentially fraudulent transactions using a trained machine learning model.
The dataset used comes from [Kaggle](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection).
""")

# Load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets8.lottiefiles.com/packages/lf20_yhTqG2.json"
lottie_hello = load_lottieurl(lottie_url)

with st.sidebar:
    st_lottie(lottie_hello, quality='high')

    st.title('Feature Guide')
    st.markdown("**step**: Time step (1 hour per step)")
    st.markdown("**type**: Type of transaction")
    st.markdown("**amount**: Transaction amount")
    st.markdown("**oldbalanceOrg**: Sender balance before transaction")
    st.markdown("**newbalanceOrig**: Sender balance after transaction")
    st.markdown("**oldbalanceDest**: Recipient balance before")
    st.markdown("**newbalanceDest**: Recipient balance after")

# User input
st.header('User Input Features')
def user_input_features():
    step = st.number_input('Step (0 to 3)', 0, 3)
    type = st.selectbox('Transaction Type', ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"))
    amount = st.number_input("Transaction Amount")
    oldbalanceOrg = st.number_input("Old Balance (Origin)")
    newbalanceOrig = st.number_input("New Balance (Origin)")
    oldbalanceDest = st.number_input("Old Balance (Destination)")
    newbalanceDest = st.number_input("New Balance (Destination)")
    
    data = {
        'step': step,
        'type': type,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Load sample data to help with one-hot encoding
fraud_raw = pd.read_csv('samp_online.csv')
fraud = fraud_raw.drop(columns=['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'])

# Combine input with existing data for correct encoding
df = pd.concat([input_df, fraud], axis=0)

# One-hot encode the 'type' column
df = pd.get_dummies(df, columns=['type'])

# Keep only the first row (user input)
df = df[:1]

# Load model and predict
if st.button("Predict"):
    load_clf = tf.keras.models.load_model('fraud.h5', compile=False)
    load_clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Ensure all columns match model input
    expected_columns = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest', 
        'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 
        'type_PAYMENT', 'type_TRANSFER'
    ]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]

    # Convert to float32 for TensorFlow
    X = df.astype(np.float32).to_numpy()

    # Predict
    y_probs = load_clf.predict(X)
    pred = tf.round(y_probs)
    pred = tf.cast(pred, tf.int32)

    st.markdown("""
        <style>[data-testid="stMetricValue"] { font-size: 25px; }</style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    if int(pred[0][0]) == 0:
        col1.metric("Prediction", value="Not Fraudulent")
    else:
        col1.metric("Prediction", value="Fraudulent Transaction")
        
    col2.metric("Confidence", value=f"{np.round(np.max(y_probs) * 100, 2)}%")
