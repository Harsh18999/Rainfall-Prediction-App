import streamlit as st
import numpy as np
import pandas as pd
import requests
from opencage.geocoder import OpenCageGeocode
from datetime import datetime
import joblib

API_KEY = "ede0c6a42bde432e9e71e01aec1c8705"
geocoder = OpenCageGeocode(API_KEY)

# Function to calculate sunshine duration
def calculate_sunshine_duration(sunrise_ts, sunset_ts):
    sunrise_time = datetime.utcfromtimestamp(sunrise_ts)
    sunset_time = datetime.utcfromtimestamp(sunset_ts)
    duration = sunset_time - sunrise_time
    return duration.total_seconds() / 3600  # Convert seconds to hours

# Function to calculate dew point
def calculate_dewpoint(temp, humidity):
    return temp - ((100 - humidity) / 5)

# Streamlit App Layout
st.set_page_config(page_title="Rainfall Prediction", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #f0f2f6;}
        .stSidebar {background-color: #e6eaf0;}
    </style>
""", unsafe_allow_html=True)

# Title
st.title("☔ Rainfall Prediction Dashboard")
st.markdown("---")

# Sidebar for user input
st.sidebar.header("📍 Location Input")
city = st.sidebar.text_input("Enter City Name")
state = st.sidebar.text_input("Enter State Name (Optional)")
country = st.sidebar.text_input("Enter Country Name")
submit = st.sidebar.button("🔍 Get Weather & Predict")

# Main layout
data_col, map_col = st.columns([2, 1])

if submit:
    if city and country:
        query = f"{city}, {state}, {country}" if state else f"{city}, {country}"
        results = geocoder.geocode(query)
        
        if results:
            latitude = results[0]['geometry']['lat']
            longitude = results[0]['geometry']['lng']
            
            with map_col:
                st.sidebar.success(f"✅ Location Found: {city}, {country}")
                st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))
            
            # Fetch Weather Data
            api_key = "82587ef327c30b6e9210054d8780a203"
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"
            model = joblib.load('model_1.pkl')
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                pressure = data['main']['pressure']
                temp_max = data['main']['temp_max']
                temperature = data['main']['temp']
                humidity = data['main']['humidity']
                wind_speed = data['wind']['speed'] * 3.6  # Convert m/s to km/h
                cloud = data['clouds']['all']
                sunshine = calculate_sunshine_duration(data['sys']['sunrise'], data['sys']['sunset'])
                dew_point = calculate_dewpoint(temperature, humidity)
                day = datetime.today().timetuple().tm_yday
                sin_day = np.sin(2 * np.pi * day / 365)
                cos_day = np.cos(2 * np.pi * day / 365)
                test_point = np.array([[pressure, temp_max, dew_point, humidity, cloud, sunshine, wind_speed, sin_day, cos_day]])
                
                # Display weather data
                with data_col:
                    st.subheader("🌦️ Current Weather Data")
                    weather_data = pd.DataFrame({
                        "Parameter": ["Temperature", "Max Temperature", "Humidity", "Pressure", "Wind Speed", "Cloud Cover", "Sunshine Duration", "Dew Point"],
                        "Value": [
                            f"{temperature}°C", f"{temp_max}°C", f"{humidity}%", f"{pressure} hPa", 
                            f"{wind_speed:.2f} km/h", f"{cloud}%", f"{sunshine:.2f} hours", f"{dew_point:.2f}°C"
                        ]
                    })
                    st.table(weather_data)
                    
                # Prediction
                prediction = model.predict_proba(test_point)
                
                # Display Prediction Results
                with data_col:
                    st.subheader("📊 Rainfall Prediction Probability")
                    st.metric(label="Chance of Rain", value=f"{prediction[0][1] * 100:.2f}%")
                    
            else:
                st.error(f"❌ Error fetching weather data: {response.status_code}")
        else:
            st.error("❌ Location not found. Please check your input.")
    else:
        st.warning("⚠️ Please enter both city and country.")
