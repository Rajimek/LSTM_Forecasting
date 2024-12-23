import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
import numpy as np

BASE_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "../data"))

def forward_fill_in_data(data, target_dates, prefix):
    """
    Forward fills missing data in a DataFrame and adds a prefix to column names.
    """
    df = pd.DataFrame(data)
    df['Dates'] = pd.to_datetime(df['Dates'])
    df.set_index('Dates', inplace=True)
    
    df = df.sort_index()
    df_filled = df.reindex(target_dates, method='ffill')
    df_filled = df_filled.add_prefix(prefix)
    
    return df_filled

RollingWindow = namedtuple('RollingWindow', ['start_date', 'end_date', 'daily_data'])

def create_rolling_windows(data_path, window_months=5):
    """
    Creates rolling windows of data with specified window size in months.
    """
    df = pd.read_csv(data_path)
    df['Dates'] = pd.to_datetime(df['Dates'])
    df = df.sort_values('Dates')
    
    columns = df.columns.tolist()
    windows_data = []
    unique_months = df['Dates'].dt.to_period('M').unique()
    
    for i in range(len(unique_months) - window_months + 1):
        start_month = unique_months[i]
        end_month = unique_months[i + window_months - 1]
      
        mask = (df['Dates'].dt.to_period('M') >= start_month) & \
               (df['Dates'].dt.to_period('M') <= end_month)
        window_df = df[mask]
        
        daily_data = [columns] + window_df.values.tolist()
        
        window = RollingWindow(
            start_date=window_df['Dates'].min(),
            end_date=window_df['Dates'].max(),
            daily_data=daily_data
        )
        
        windows_data.append(window)
    
    return windows_data

def download_data(data, file_name):
    """
    Saves various data structures to CSV files.
    """
    file_path = os.path.join(BASE_DIR, f'{file_name}.csv')
    
    if isinstance(data, dict):
        df = pd.DataFrame([data])  # Convert single dictionary to DataFrame
    elif isinstance(data, list):
        if len(data) == 0:
            df = pd.DataFrame()
        elif isinstance(data[0], list):
            if any(isinstance(seq[0], list) for seq in data):
                all_data = []
                for sequence in data:
                    all_data.extend(sequence[1:])
                df = pd.DataFrame(all_data, columns=data[0][0])
            else:
                df = pd.DataFrame(data[1:], columns=data[0])
        elif isinstance(data[0], RollingWindow):
            df = pd.DataFrame({
                'start_date': [window.start_date for window in data],
                'end_date': [window.end_date for window in data],
                'daily_data': [window.daily_data for window in data]
            })
        else:
            df = pd.DataFrame({'value': data})
    elif isinstance(data, RollingWindow):
        df = pd.DataFrame({
            'start_date': [data.start_date],
            'end_date': [data.end_date],
            'daily_data': [data.daily_data]
        })
    else:
        df = data.reset_index()
    
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

def create_price_labels(windows_data, prediction_type='monthly', threshold_pct=0.02):
    """
    Creates binary labels for price predictions based on percentage change.
    
    Args:
        windows_data (List[RollingWindow]): List of RollingWindow objects
        prediction_type (str): Either 'weekly' or 'monthly'
        threshold_pct (float): Minimum percentage change to consider a significant move
                             (e.g., 0.02 means 2% move required)
        
    Returns:
        List[RollingWindow]: Original windows with labels added to daily_data headers
    """
    labeled_windows = []
    px_last_idx = windows_data[0].daily_data[0].index('SPX_PX_LAST')
    
    for i in range(len(windows_data) - 1):
        current_window = windows_data[i]
        next_window = windows_data[i + 1]
        
        current_data = current_window.daily_data
        current_price = float(current_data[-1][px_last_idx])  # Last price in current window
        next_price = float(next_window.daily_data[1][px_last_idx])  # First price in next window
        
        # Calculate percentage change
        pct_change = (next_price - current_price) / current_price
        
        # Skip if change is within the noise threshold
        if abs(pct_change) < threshold_pct:
            continue
        
        # Create binary label based on direction
        label = 1 if pct_change >= threshold_pct else 0
        
        # Add label to the headers and each row
        headers = current_data[0] + ['label']
        labeled_rows = [headers]
        
        # Add label to each row of data
        for row in current_data[1:]:
            labeled_rows.append(row + [label])
        
        # Create new RollingWindow with labeled data
        labeled_window = RollingWindow(
            start_date=current_window.start_date,
            end_date=current_window.end_date,
            daily_data=labeled_rows
        )
        
        labeled_windows.append(labeled_window)
    
    return labeled_windows

def save_sequences_to_csv(sequences, file_name):
    """
    Saves sequences with separated features and labels to CSV.
    """
    rows = []
    for seq in sequences:
        # Extract features (all columns except the label) and label from daily_data
        headers = seq.daily_data[0]
        features = [row[:-1] for row in seq.daily_data[1:]]  # All columns except label
        label = seq.daily_data[1][-1]  # Get label from first row (all rows have same label)
        
        rows.append({
            'features': str(features),
            'label': label
        })
    
    df = pd.DataFrame(rows)
    download_data(df, file_name)

def print_sequence_statistics(sequences, window_size, prediction_type):
    """
    Print statistics about the labeled sequences.
    """
    total_sequences = len(sequences)
    if total_sequences == 0:
        print(f"\nNo {window_size} {prediction_type} sequences met the threshold criteria")
        return
        
    # Get label from the first row of daily_data (excluding header)
    labels = [seq.daily_data[1][-1] for seq in sequences]
    
    # Calculate percentage changes using the first and last prices from each sequence
    px_last_idx = sequences[0].daily_data[0].index('SPX_PX_LAST')
    pct_changes = []
    for seq in sequences:
        first_price = float(seq.daily_data[1][px_last_idx])
        last_price = float(seq.daily_data[-1][px_last_idx])
        pct_change = (last_price - first_price) / first_price
        pct_changes.append(pct_change)
    
    upward_moves = sum(labels)
    downward_moves = len(labels) - upward_moves
    
    print(f"\n{window_size} {prediction_type.capitalize()} Label Statistics:")
    print(f"Total sequences: {total_sequences}")
    print(f"Upward moves (1): {upward_moves} ({upward_moves/total_sequences*100:.1f}%)")
    print(f"Downward moves (0): {downward_moves} ({downward_moves/total_sequences*100:.1f}%)")
    print(f"Average upward move: {np.mean([pc for l, pc in zip(labels, pct_changes) if l == 1])*100:.1f}%")
    print(f"Average downward move: {np.mean([pc for l, pc in zip(labels, pct_changes) if l == 0])*100:.1f}%")
    print(f"Max upward move: {max(pct_changes)*100:.1f}%")
    print(f"Max downward move: {min(pct_changes)*100:.1f}%")

def plot_correlation_matrix(data):
    """
    Creates and plots a correlation matrix heatmap for the metrics features.
    """
    correlation_matrix = data.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, 
                annot=True,  
                cmap='coolwarm',  
                center=0,  
                square=True,  
                fmt='.2f',  
                linewidths=0.5)  
    
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load and process data files
    historical_prices_file = os.path.join(BASE_DIR, 'bloomberg/historical_prices_spx.csv')
    con_conf_file = os.path.join(BASE_DIR, 'bloomberg/index_metrics_CONCCONF.csv')
    napm_pmi_file = os.path.join(BASE_DIR, 'bloomberg/index_metrics_NAPMPMI.csv')
    spx_index_file = os.path.join(BASE_DIR, 'bloomberg/index_metrics_SPX.csv')
    us_10yr_file = os.path.join(BASE_DIR, 'bloomberg/index_metrics_USGG10YR.csv')
    vix_file = os.path.join(BASE_DIR, 'bloomberg/index_metrics_VIX.csv')

    # Read and process historical prices
    historical_prices_spx = pd.read_csv(historical_prices_file)
    historical_prices_spx['Dates'] = pd.to_datetime(historical_prices_spx['Dates'])
    historical_prices_spx.set_index('Dates', inplace=True)
    historical_prices_spx = historical_prices_spx.sort_index()
    
    target_dates = historical_prices_spx.index
    
    # Read and forward fill all data
    data_files = {
        'con_conf': (pd.read_csv(con_conf_file), 'CONCCONF_'),
        'napm_pmi': (pd.read_csv(napm_pmi_file), 'NAPMPMI_'),
        'spx_index': (pd.read_csv(spx_index_file), 'SPXINDEX_'),
        'us_10yr': (pd.read_csv(us_10yr_file), 'USGG10YR_'),
        'vix': (pd.read_csv(vix_file), 'VIX_')
    }
    
    filled_data = {name: forward_fill_in_data(data, target_dates, prefix) 
                  for name, (data, prefix) in data_files.items()}
    
    # Merge all data
    merged_data = pd.concat([
        historical_prices_spx.add_prefix('SPX_'),
        *filled_data.values()
    ], axis=1)

    download_data(merged_data, 'forward_filled_merged_data')
    merged_data_file = os.path.join(BASE_DIR, 'forward_filled_merged_data.csv')

    # Create sequences
    five_month_sequences = create_rolling_windows(merged_data_file, window_months=5)
    three_quarter_sequences = create_rolling_windows(merged_data_file, window_months=9)

    # Fix: Access daily_data from the first RollingWindow object
    first_five_month_sequence = five_month_sequences[0].daily_data
    first_three_quarter_sequence = three_quarter_sequences[0].daily_data

    download_data(first_five_month_sequence, 'first_five_month_sequence')
    download_data(first_three_quarter_sequence, 'first_three_quarter_sequence')

    # Create labeled sequences with different thresholds for weekly and monthly
    five_month_monthly = create_price_labels(
        five_month_sequences, 
        prediction_type='monthly',
        threshold_pct=0.03  # 3% threshold for monthly
    )
    
    five_month_weekly = create_price_labels(
        five_month_sequences, 
        prediction_type='weekly',
        threshold_pct=0.02  # 2% threshold for weekly
    )
    
    three_quarter_monthly = create_price_labels(
        three_quarter_sequences, 
        prediction_type='monthly',
        threshold_pct=0.03
    )
    
    three_quarter_weekly = create_price_labels(
        three_quarter_sequences, 
        prediction_type='weekly',
        threshold_pct=0.02
    )

    # Save sequences and print statistics
    save_sequences_to_csv(five_month_monthly, 'five_month_monthly_predictions')
    save_sequences_to_csv(five_month_weekly, 'five_month_weekly_predictions')
    save_sequences_to_csv(three_quarter_monthly, 'three_quarter_monthly_predictions')
    save_sequences_to_csv(three_quarter_weekly, 'three_quarter_weekly_predictions')

    # Print statistics for all sequence types
    print_sequence_statistics(five_month_monthly, "5-month", "monthly")
    print_sequence_statistics(five_month_weekly, "5-month", "weekly")
    print_sequence_statistics(three_quarter_monthly, "9-month", "monthly")
    print_sequence_statistics(three_quarter_weekly, "9-month", "weekly")

    # After creating five_month_monthly predictions
    first_sequence = five_month_monthly[0]  # Get first sequence
    
    # Extract features and label
    first_sequence_features = first_sequence.daily_data[1:-1]  # All rows except header and last row
    first_sequence_label = first_sequence.daily_data[1][-1]    # Label from first data row
    
    # Save to CSV
    download_data(
        {
            'features': str(first_sequence_features),
            'label': first_sequence_label
        }, 
        'first_five_month_monthly_sequence'
    )
    
    # Print for verification
    print("\nFirst Five Month Monthly Sequence:")
    print(f"Features: {first_sequence_features}")
    print(f"Label: {first_sequence_label}")

    # print("\nGenerating correlation matrix...")
    # plot_correlation_matrix(merged_data)