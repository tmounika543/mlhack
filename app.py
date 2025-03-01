import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

# Plotting the COVID-19 data for USA
labels = ["Total Cases", "Active Cases", "Recovered", "Deaths"]
values = [data["cases"], data["active"], data["recovered"], data["deaths"]]

plt.figure(figsize=(8,5))
plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("COVID-19 Data for USA")

# Streamlit setup
st.title("COVID-19 Cases Prediction in USA")
st.write("Displaying current COVID-19 data for the USA")

# Show the bar plot in Streamlit
st.pyplot(plt)

# Generate random historical data for demonstration
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)

df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

# Plot historical cases and deaths
plt.figure(figsize=(10, 6))
plt.plot(df_historical["day"], df_historical["cases"], label="Cases", color="blue", marker="o")
plt.plot(df_historical["day"], df_historical["deaths"], label="Deaths", color="red", marker="x")
plt.xlabel("Day")
plt.ylabel("Count")
plt.title("COVID-19 Historical Cases and Deaths (Last 30 Days)")
plt.legend()

# Show the historical data plot in Streamlit
st.pyplot(plt)

# Predicting the next day's cases using Linear Regression
X = df_historical[["day"]]
y = df_historical["cases"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict next day's cases
next_day = np.array([[31]])
predicted_cases = model.predict(next_day)
st.write(f"Predicted cases for Day 31: {int(predicted_cases[0])}")

# User input for prediction
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

if st.button("Predict"):
    prediction = model.predict([[day_input]])
    st.write(f"Predicted cases for day {day_input}: {int(prediction[0])}")
