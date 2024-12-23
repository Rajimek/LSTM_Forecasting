import os
import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ---------------------------------------------------------
# Load and Process the Dataset
# ---------------------------------------------------------
BASE_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "../data"))
OUTPUT_DIR = os.path.join(BASE_DIR, "arima")
os.makedirs(OUTPUT_DIR, exist_ok=True)

file_path = os.path.join(BASE_DIR, 'forward_filled_merged_data.csv')
spx_data = pd.read_csv(file_path)

spx_data['Dates'] = pd.to_datetime(spx_data['Dates'])
spx_data.set_index('Dates', inplace=True)
spx_data = spx_data[['SPX_PX_LAST']]  

train_data, test_data = train_test_split(spx_data, test_size=0.2, shuffle=False)
train_data.index = spx_data.index[:len(train_data)]  
test_data.index = spx_data.index[len(train_data):]  

# ---------------------------------------------------------
# Auto ARIMA Model
# ---------------------------------------------------------
model = auto_arima(train_data, seasonal=False, trace=True, suppress_warnings=True, stepwise=True)
model.fit(train_data)

# -------------------------------------2--------------------
# Make Predictions
# ---------------------------------------------------------
forecast = model.predict(n_periods=len(test_data))

forecast.index = test_data.index[:len(forecast)]
test_data['Forecast'] = forecast

# ---------------------------------------------------------
# Calculate Accuracy Metrics
# ---------------------------------------------------------
test_data_path = os.path.join(OUTPUT_DIR, 'test_data.csv')
test_data.to_csv(test_data_path)

mae = mean_absolute_error(test_data['SPX_PX_LAST'], test_data['Forecast'])
mse = mean_squared_error(test_data['SPX_PX_LAST'], test_data['Forecast'])
rmse = np.sqrt(mse)

mean_actual = test_data['SPX_PX_LAST'].mean()  
accuracy_percentage = (1 - mae / mean_actual) * 100  

accuracy_path = os.path.join(OUTPUT_DIR, 'accuracy_metrics.txt')
with open(accuracy_path, 'w') as f:
    f.write(f"Mean Absolute Error (MAE): {mae}\n")
    f.write(f"Mean Squared Error (MSE): {mse}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
    f.write(f"Accuracy Percentage: {accuracy_percentage:.2f}%\n")

print(f"Accuracy metrics saved to {accuracy_path}")
print(f"Accuracy Percentage: {accuracy_percentage:.2f}%")

# ---------------------------------------------------------
# Plot Results
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(train_data, label="Training Data")
plt.plot(test_data['SPX_PX_LAST'], label="Testing Data", color="blue")  
plt.plot(test_data['Forecast'], label="Forecast", color="orange", linestyle="--")
plt.legend()
plt.title("SPX Price Forecast using Auto ARIMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plot_path = os.path.join(OUTPUT_DIR, 'spx_forecast_plot.png')
plt.savefig(plot_path)
plt.close()

print(f"Plot saved to {plot_path}")


