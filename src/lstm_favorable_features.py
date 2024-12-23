import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------
# Load and Preprocess Data
# ---------------------------------------------------------
BASE_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "../data"))
OUTPUT_DIR = os.path.join(BASE_DIR, "presentation_features_favorable")
os.makedirs(OUTPUT_DIR, exist_ok=True)

file_path = os.path.join(BASE_DIR, 'forward_filled_merged_data.csv')
forward_filled_data = pd.read_csv(file_path)

forward_filled_data['Dates'] = pd.to_datetime(forward_filled_data['Dates'])
forward_filled_data.set_index('Dates', inplace=True)

price_column = "SPX_PX_LAST"
correlation_matrix = forward_filled_data.corr()
price_correlations = correlation_matrix[price_column]
significant_features = price_correlations[price_correlations.abs() > 0.5].index.tolist()
significant_features.remove(price_column)
2
features_data = forward_filled_data[significant_features]
target_data = forward_filled_data[[price_column]]


train_size = int(len(features_data) * 0.6)
val_size = int(len(features_data) * 0.2)

train_features = features_data[:train_size]
val_features = features_data[train_size:train_size + val_size]
test_features = features_data[train_size + val_size:]

train_target = target_data[:train_size]
val_target = target_data[train_size:train_size + val_size]
test_target = target_data[train_size + val_size:]

scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

train_features_scaled = scaler_features.fit_transform(train_features)
train_target_scaled = scaler_target.fit_transform(train_target)

val_features_scaled = scaler_features.transform(val_features)
val_target_scaled = scaler_target.transform(val_target)
test_features_scaled = scaler_features.transform(test_features)
test_target_scaled = scaler_target.transform(test_target)

train_data = np.hstack((train_features_scaled, train_target_scaled))
val_data = np.hstack((val_features_scaled, val_target_scaled))
test_data = np.hstack((test_features_scaled, test_target_scaled))

# ---------------------------------------------------------
# Prepare Data for LSTM
# ---------------------------------------------------------
window_size = 216

def create_rolling_windows_multifeature(data, target_column_index, window_size=216):
    """Create sequences for multivariate time series."""
    features, targets = [], []
    for i in range(len(data) - window_size):
        features.append(data[i:i + window_size])
        targets.append(data[i + window_size, target_column_index])
    return np.array(features), np.array(targets)

target_column_index = len(significant_features)  

X_train, y_train = create_rolling_windows_multifeature(train_data, target_column_index, window_size)
X_val, y_val = create_rolling_windows_multifeature(val_data, target_column_index, window_size)
X_test, y_test = create_rolling_windows_multifeature(test_data, target_column_index, window_size)

print(f"Training sequences shape: {X_train.shape}")
print(f"Validation sequences shape: {X_val.shape}")
print(f"Testing sequences shape: {X_test.shape}")

# ---------------------------------------------------------
# Build and Train the LSTM Model
# ---------------------------------------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    min_delta=1e-4
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[early_stopping]
)

# ---------------------------------------------------------
# Evaluate the Model
# ---------------------------------------------------------
y_pred = model.predict(X_test)

y_pred_rescaled = scaler_target.inverse_transform(y_pred)
y_test_rescaled = scaler_target.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
mean_actual = np.mean(y_test_rescaled)
accuracy = (1 - mae / mean_actual) * 100

print(f"\nModel Performance Metrics:")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Average Actual Price: ${mean_actual:.2f}")

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

test_dates = forward_filled_data.index[train_size + val_size + window_size:][:len(y_test_rescaled)]
forecast_df = pd.DataFrame({
    'Actual': y_test_rescaled.flatten(),
    'Forecast': y_pred_rescaled.flatten()
}, index=test_dates)

plt.figure(figsize=(14, 7))
plt.plot(forecast_df.index, forecast_df['Actual'], label='Actual', linewidth=2)
plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', linewidth=2, linestyle='--')
plt.title('LSTM Predictions vs Actual')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()

predictions_plot_path = os.path.join(OUTPUT_DIR, "predictions_vs_actual.png")
plt.savefig(predictions_plot_path)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

loss_plot_path = os.path.join(OUTPUT_DIR, "training_validation_loss.png")
plt.savefig(loss_plot_path)
plt.show()