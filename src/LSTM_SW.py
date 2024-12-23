import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Load and Process the Dataset
# ---------------------------------------------------------
BASE_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "../data"))
OUTPUT_DIR = os.path.join(BASE_DIR, "lstm_70_15_15")

file_path = os.path.join(BASE_DIR, 'forward_filled_merged_data.csv')
spx_data = pd.read_csv(file_path)
spx_data['Dates'] = pd.to_datetime(spx_data['Dates'])
spx_data.set_index('Dates', inplace=True)

if 'SPX_PX_LAST' not in spx_data.columns:
    raise ValueError("The column 'SPX_PX_LAST' is not found in the dataset.")

spx_prices = spx_data['SPX_PX_LAST'].values.reshape(-1, 1)

# ---------------------------------------------------------
# Split and scale data
# ---------------------------------------------------------
train_size = int(len(spx_prices) * 0.6)
val_size = int(len(spx_prices) * 0.2)
test_size = len(spx_prices) - train_size - val_size

train_data = spx_prices[:train_size]
val_data = spx_prices[train_size:train_size + val_size]
test_data = spx_prices[train_size + val_size:]

scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)
val_data_scaled = scaler.transform(val_data)
test_data_scaled = scaler.transform(test_data)

# ---------------------------------------------------------
# Create Rolling Windows for Forecasting
# ---------------------------------------------------------
window_size = 216

def create_rolling_windows_forecast(data, window_size=216):
    """
    Create sequences of window_size days of data to predict the next day.
    """
    features, targets = [], []
    for i in range(len(data) - window_size):
        features.append(data[i:i + window_size])
        targets.append(data[i + window_size])  
    return np.array(features), np.array(targets)

X_train, y_train = create_rolling_windows_forecast(train_data_scaled, window_size)
X_val, y_val = create_rolling_windows_forecast(val_data_scaled, window_size)
X_test, y_test = create_rolling_windows_forecast(test_data_scaled, window_size)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"Training sequences: {X_train.shape}")
print(f"Validation sequences: {X_val.shape}")
print(f"Testing sequences: {X_test.shape}")

# ---------------------------------------------------------
# Build and Train the LSTM Model
# ---------------------------------------------------------
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    min_delta=1e-4
)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(window_size, 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[early_stopping]
)

# ---------------------------------------------------------
# Forecast and Evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test)

y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100
mean_actual = np.mean(y_test_rescaled)
accuracy_percentage = (1 - mae / mean_actual) * 100

print("\nModel Performance Metrics:")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Accuracy: {accuracy_percentage:.2f}%")
print(f"Average Actual Price: ${mean_actual:.2f}")

# ---------------------------------------------------------
# Align Forecast Dates for Plotting
# ---------------------------------------------------------
forecast_start_index = train_size + val_size + window_size
forecast_end_index = forecast_start_index + len(y_test_rescaled)
forecast_dates = spx_data.index[forecast_start_index:forecast_end_index]

forecast_df = pd.DataFrame({
    'Actual': y_test_rescaled.flatten(),
    'Forecast': y_pred_rescaled.flatten()
}, index=forecast_dates)

OUTPUT_DIR = os.path.join(BASE_DIR, "presentation_final")
os.makedirs(OUTPUT_DIR, exist_ok=True)

forecast_data_path = os.path.join(OUTPUT_DIR, 'spx_with_forecast.csv')
forecast_df.to_csv(forecast_data_path)
print(f"Forecast data saved to {forecast_data_path}")

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
plt.figure(figsize=(14, 7))
plt.plot(forecast_dates, forecast_df['Actual'], 
         label='Actual', linewidth=2, color='blue')
plt.plot(forecast_dates, forecast_df['Forecast'], 
         label='Forecast', linewidth=2, color='orange')

plt.title('S&P 500 Price Prediction (Test Period)')
plt.xlabel('Date')
plt.ylabel('SPX Prices ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()

forecast_plot_path = os.path.join(OUTPUT_DIR, 'forecast_results.png')
plt.savefig(forecast_plot_path)
plt.show()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


loss_plot_path = os.path.join(OUTPUT_DIR, 'training_validation_loss.png')
plt.savefig(loss_plot_path)
plt.show()
