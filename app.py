import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Fetch COVID-19 data for the USA
url = "https://disease.sh/v3/covid-19/countries/usa"
r = requests.get(url)
data = r.json()

# Extract relevant fields
covid_data = {
    "cases": data["cases"],
    "todayCases": data["todayCases"],
    "deaths": data["deaths"],
    "todayDeaths": data["todayDeaths"],
    "recovered": data["recovered"],
    "active": data["active"],
    "critical": data["critical"],
    "casesPerMillion": data["casesPerOneMillion"],
    "deathsPerMillion": data["deathsPerOneMillion"],
}

# Convert to Pandas DataFrame
df = pd.DataFrame([covid_data])

# Generate random historical data for demonstration
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)

df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

# Create a Linear Regression model
X = df_historical[["day"]]
y = df_historical["cases"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("COVID-19 Cases Prediction-in USA")
st.write("Predicting COVID-19 cases for the next day based on historical data.")

# User Input for day number
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

# Button to trigger prediction
if st.button("Predict"):
    # Predict cases for the input day
    prediction = model.predict([[day_input]])
    st.write(f"Predicted cases for day {day_input}: {int(prediction[0])}")
    
    # Calculate Mean Squared Error and R² Score on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Display MSE and R² score
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R² Score: {r2}")
    
    # Predict for future days (for plotting the trend)
    future_days = np.array(range(31, day_input + 1))  # Generate future days
    future_days = future_days.reshape(-1, 1)  # Reshaping for prediction
    
    # Predict future cases
    future_predictions = model.predict(future_days)
    
    # Plotting historical and predicted cases
    plt.figure(figsize=(10, 6))
    plt.plot(df_historical["day"], df_historical["cases"], label="Historical Cases", color="blue", marker="o")
    plt.plot(future_days, future_predictions, label="Predicted Cases", color="green", linestyle="--", marker="x")
    plt.xlabel("Day")
    plt.ylabel("Case Count")
    plt.title(f"COVID-19 Predicted Cases for Day {day_input}")
    plt.legend()
    
    # Show the graph in Streamlit
    st.pyplot(plt)
