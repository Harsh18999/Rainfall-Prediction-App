import streamlit as st
import numpy as np
import pandas as pd
import requests
from opencage.geocoder import OpenCageGeocode
from datetime import datetime
import joblib

# Replace 'YOUR_API_KEY' with your actual OpenCage API key
API_KEY = 'ede0c6a42bde432e9e71e01aec1c8705'
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

# Streamlit App Title
st.title("☔ Rainfall Prediction App")
st.write("Predict the probability of rainfall in the next 24 hours.")

# Sidebar for user input
st.sidebar.header("📍 Location Input")
city = st.sidebar.text_input("Enter City Name")
state = st.sidebar.text_input("Enter State Name (Optional)")
country = st.sidebar.text_input("Enter Country Name")
submit = st.sidebar.button("Get Weather & Predict")

if submit:
    if city and country:
        query = f"{city}, {state}, {country}" if state else f"{city}, {country}"
        results = geocoder.geocode(query)
        
        if results:
            latitude = results[0]['geometry']['lat']
            longitude = results[0]['geometry']['lng']
            
            st.sidebar.success(f"✅ Location Found: {city}, {country}")
            st.sidebar.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))
            
            # Fetch Weather Data
            api_key = "82587ef327c30b6e9210054d8780a203"  # Replace with your actual API key
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"
            model = joblib.load('model_1.pkl', mmap_mode=None)
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
                st.subheader("🌦️ Current Weather Data")
                st.write(f"**Temperature:** {temperature}°C")
                st.write(f"**Max Temperature:** {temp_max}°C")
                st.write(f"**Humidity:** {humidity}%")
                st.write(f"**Pressure:** {pressure} hPa")
                st.write(f"**Wind Speed:** {wind_speed:.2f} km/h")
                st.write(f"**Cloud Cover:** {cloud}%")
                st.write(f"**Sunshine Duration:** {sunshine:.2f} hours")
                st.write(f"**Dew Point:** {dew_point:.2f}°C")
                
                # Prediction
                prediction = model.predict_proba(test_point)
                
                # Display Prediction Results
                st.subheader("📊 Rainfall Prediction Probability")
                st.write(f"**Probability of Rain in Next 24 Hours:** {prediction[0][1] * 100:.2f}%")
                
            else:
                st.error(f"❌ Error fetching weather data: {response.status_code}")
        else:
            st.error("❌ Location not found. Please check your input.")
    else:
        st.warning("⚠️ Please enter both city and country.")
