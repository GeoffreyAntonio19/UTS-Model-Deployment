import streamlit as st
import pandas as pd
import joblib

# Load model, preprocessor, dan label encoder
model = joblib.load("xgb_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Fungsi prediksi
def predict_new_data(new_data: pd.DataFrame):
    transformed_data = preprocessor.transform(new_data)
    pred_encoded = model.predict(transformed_data)
    pred_decoded = label_encoder.inverse_transform(pred_encoded)
    return pred_decoded[0]

# Judul aplikasi
st.title("Hotel Booking Cancellation Prediction")

# Input dari user
st.header("Input Booking Details")

with st.form(key="booking_form"):
    no_of_adults = st.number_input("Number of Adults", min_value=1, value=2)
    no_of_children = st.number_input("Number of Children", min_value=0, value=0)
    no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=2)
    no_of_week_nights = st.number_input("Week Nights", min_value=0, value=2)
    type_of_meal_plan = st.selectbox("Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
    required_car_parking_space = st.selectbox("Requires Parking Space", [0, 1])
    room_type_reserved = st.selectbox("Room Type", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
    lead_time = st.number_input("Lead Time", min_value=0, value=35)
    arrival_year = st.selectbox("Arrival Year", [2017, 2018])
    arrival_month = st.selectbox("Arrival Month", list(range(1, 13)))
    arrival_date = st.selectbox("Arrival Date", list(range(1, 32)))
    market_segment_type = st.selectbox("Market Segment", ["Online", "Offline", "Corporate", "Complementary", "Aviation", "Other"])
    repeated_guest = st.selectbox("Repeated Guest", [0, 1])
    no_of_previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
    no_of_previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, value=0)
    avg_price_per_room = st.number_input("Average Price Per Room", min_value=0.0, value=100.0)
    no_of_special_requests = st.number_input("Special Requests", min_value=0, value=1)

    submit_button = st.form_submit_button("Predict")

# Prediksi dan hasilnya
if submit_button:
    input_data = pd.DataFrame([{
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'type_of_meal_plan': type_of_meal_plan,
        'required_car_parking_space': required_car_parking_space,
        'room_type_reserved': room_type_reserved,
        'lead_time': lead_time,
        'arrival_year': arrival_year,
        'arrival_month': arrival_month,
        'arrival_date': arrival_date,
        'market_segment_type': market_segment_type,
        'repeated_guest': repeated_guest,
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'avg_price_per_room': avg_price_per_room,
        'no_of_special_requests': no_of_special_requests
    }])

    prediction = predict_new_data(input_data)
    st.success(f"Prediction: {prediction}")
