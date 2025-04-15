


# Description: This script contains the functions used in the Flexa Challenge submission.



# --------- Task 1 Libraries --------- #
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error
from scipy.signal import periodogram

# --------- Task 2 Libraries --------- #
import cvxpy as cp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.ndimage import gaussian_filter1d
import random

# ---------- Task 1 Functions ---------- #

pool_color = cm.rainbow(np.linspace(0, 1, 14))


def reformating_df_and_target(df):
    """
    Reformat the DataFrame and create the target variables "production", "consumption", and "net_energy"

    Args:
    df (pd.DataFrame): DataFrame containing the time series data

    Returns:
    df (pd.DataFrame): DataFrame containing the time series data with the target variables
    """

    # Create new columns "production" and "consumption"
    df['production'] = df.apply(lambda row: row['target'] if row['is_consumption'] == 0 else 0, axis=1)
    df['consumption'] = df.apply(lambda row: row['target'] if row['is_consumption'] == 1 else 0, axis=1)

    # Create "net_energy" Column
    df['net_energy'] = df['production'] - df['consumption']

    # Drop "target" and "is_consumption" Columns
    df.drop(['target', 'is_consumption'], axis=1, inplace=True)

    # Group by 'pool' and 'datetime_utc' and sum the other variables
    df = df.groupby(['pool', 'datetime_utc']).sum().reset_index()

    return df



def check_time_series_continuity(df):
    """
    Check if the time series is continuous, identify missing timestamps
    and interpolate the missing values if required
    
    Args:
    df (pd.DataFrame): DataFrame containing the time series data

    Returns:
    continuity (bool): True if the time series is continuous, False otherwise
    missing (pd.DatetimeIndex): DatetimeIndex containing the missing timestamps
    df (pd.DataFrame): DataFrame containing the time series data with interpolated values
    """

    full_time_range = pd.date_range(start=df['datetime_utc'].min(), 
                                    end=df['datetime_utc'].max(), 
                                    freq='h')
    
    # Check if the generated time range matches the actual timestamps
    actual_time_range = pd.DatetimeIndex(df['datetime_utc'].sort_values())
    continuity = full_time_range.equals(actual_time_range)
    missing = full_time_range.difference(actual_time_range)

    if not continuity:
        df = df.set_index('datetime_utc').reindex(full_time_range).interpolate().reset_index()
        df.rename(columns={'index': 'datetime_utc'}, inplace=True)

    return continuity, missing, df



def creating_time_dummy_variables(df):
    """
    Create additional time-related features

    Args:
    df (pd.DataFrame): DataFrame containing the time series data

    Returns:
    df (pd.DataFrame): DataFrame containing the time series data with additional features
    """

    # Extract the date from 'datetime_utc'
    df['date'] = df['datetime_utc'].dt.date

    # Daily time-related features
    df['hour_of_day'] = df['datetime_utc'].dt.hour

    # Weekly time-related features
    df['day_of_week'] = df['datetime_utc'].dt.dayofweek
    df['hour_of_week'] = df['day_of_week'] * 24 + df['hour_of_day']

    # Monthly time-related features
    df['day_of_month'] = df['datetime_utc'].dt.day

    # Yearly time-related features
    df['hour_of_year'] = df['datetime_utc'].dt.dayofyear * 24 + df['hour_of_day'] - 24
    df['day_of_year'] = df['datetime_utc'].dt.dayofyear
    df['week_of_year'] = df['datetime_utc'].dt.isocalendar().week
    df['month_of_year'] = df['datetime_utc'].dt.month
    
    # Long-term time-related features
    df['year'] = df['datetime_utc'].dt.isocalendar().year

    # Combinations of time-related features
    df['year_week'] = (df['year'].astype(str) + '-W' + df['week_of_year'].astype(str)).astype('category')

    return df



def plot_seasonalities(df, pool):
    """
    Plot the daily, weekly, and yearly seasonalities for Production and Consumption.

    Args:
    df (pd.DataFrame): DataFrame containing the time series data of a specific pool
    pool (int): Pool number
    """

    # Create a plot
    plt.figure(figsize=(15, 10))
    
    df = df[df["pool"]==pool].copy()

    # Daily Seasonalities
    for variable, subplot, title in zip(["production", "consumption", "net_energy"], [1, 2, 3], ['Production', 'Consumption', 'Net_Energy']):
        plt.subplot(3, 3, subplot)
        # Pivot the DataFrame to have hours as columns and dates as rows
        pivot_table = df.pivot(index='date', columns='hour_of_day', values=variable).copy()
        # Plot each row (date) as a separate line on the same plot
        for date in pivot_table.index:
            plt.plot(pivot_table.columns, pivot_table.loc[date], alpha=0.7, lw=0.1, color=pool_color[pool])
        # Add labels and legend
        plt.xlabel('Hour of the Day')
        plt.ylabel(f"{title}")
        plt.title(f'{title} by Hour of the Day')
        plt.xticks(np.linspace(0,24, 4))  # Ensure x-axis has ticks for each day of the week

    # Weekly Seasonalities
    for variable, subplot, title in zip(["production", "consumption", "net_energy"], [4, 5, 6], ['Production', 'Consumption', 'Net_Energy']):
        plt.subplot(3, 3, subplot)
        # Pivot the DataFrame to have hours as columns and dates as rows
        pivot_table = df.pivot(index='year_week', columns='hour_of_week', values=variable).copy()
        # Plot each row (date) as a separate line on the same plot
        for date in pivot_table.index:
            plt.plot(pivot_table.columns, pivot_table.loc[date], alpha=0.7, lw=0.1, color=pool_color[pool])
        # Add labels and legend
        plt.xlabel('Hour of the Week')
        plt.ylabel(f"{title}")
        plt.title(f'{title} by Hour of the Week')
        plt.xticks(np.linspace(0,24*7, 8), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", ""])  # Ensure x-axis has ticks for each day of the week

    # Yearly Seasonalities
    for variable, subplot, title in zip(["production", "consumption", "net_energy"], [7, 8, 9], ['Production', 'Consumption', 'Net_Energy']):
        plt.subplot(3, 3, subplot)
        # Pivot the DataFrame to have hours as columns and dates as rows
        pivot_table = df.pivot(index='year', columns='hour_of_year', values=variable).copy()
        # Plot each row (date) as a separate line on the same plot
        for date in pivot_table.index:
            plt.plot(pivot_table.columns, pivot_table.loc[date], alpha=0.7, lw=0.1, color=pool_color[pool])
        # Add labels and legend
        plt.xlabel('Hour of the year')
        plt.ylabel(f"{title}")
        plt.title(f'{title} by Hour of the year')
        plt.xticks(np.linspace(0,24*365, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", ""])  # Ensure x-axis has ticks for each day of the week

    plt.tight_layout()

    # Display the plot
    plt.show()



def peridograms(df, pool):
    """
    Plot the periodograms for Production and Consumption.
    
    Args:
    df (pd.DataFrame): DataFrame containing the time series data of a specific pool
    pool (int): Pool number
    """

    df = df[df["pool"]==pool].copy()
    
    # Calculate the frequency and power for Production and Consumption
    freqs_production, power_production = periodogram(df["production"])
    freqs_consumption, power_consumption = periodogram(df["consumption"])

    plt.figure(figsize=(15, 5))

    for subplot, freqs, power, title in zip([1, 2], [freqs_production, freqs_consumption], [power_production, power_consumption], ['Production', 'Consumption']):
        plt.subplot(2, 1, subplot)

        # Plot the periodogram
        plt.plot(freqs, power, color=pool_color[pool], alpha=0.7)

        # Half-day, daily, and yearly seasonality lines
        plt.axvline(1/12, color='yellow', linestyle='--', label='Half-day Seasonality', alpha=0.7)
        plt.axvline(1/24, color='red', linestyle='--', label='Daily Seasonality', alpha=0.7)
        plt.axvline(1/(24*365), color='cyan', linestyle='--', label='Yearly Seasonality', alpha=0.7)

        # Add axis, labels and legend
        plt.xlim(0, 0.1)
        plt.xticks(np.linspace(0, 0.1, 5))
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Power', fontsize=12)
        plt.title(f'{title} Seasonalities')
        plt.legend()

    plt.tight_layout()

    plt.show()



def net_energy_violon_plot(df):
    """
    Plot a violin plot of the net energy for each pool.
    
    Args:
    df (pd.DataFrame): DataFrame containing the time series data
    """
    plt.figure(figsize=(15, 5))

    # Iterate over each pool
    sns.violinplot(x='pool', y='net_energy', hue='pool', data=df, palette='flare', legend=False)

    # Add a title and labels
    plt.title('Violin plot of net energy for each pool')
    plt.xlabel('Pool', fontsize=12)
    plt.ylabel('Net energy', fontsize=12)

    plt.tight_layout()

    plt.show()



def correlation_matrix_plot(df):
    """
    Plot the linear correlation matrix for the numerical columns in the DataFrame.
    
    Args:
    df (pd.DataFrame): DataFrame containing the time series data
    """
    # Select only numerical columns
    ldf = df.select_dtypes(include=[np.number]) 
    
    # Create the correlation matrix
    corr_mat = ldf.corr().round(2)

    # Create a mask for the upper triangle
    _, ax = plt.subplots(figsize=(15,5))
    mask = np.triu(np.ones_like(corr_mat, dtype=np.bool))
    mask = mask[1:,:-1]

    # Plot the correlation matrix
    corr = corr_mat.iloc[1:,:-1].copy()
    sns.heatmap(corr,mask=mask,vmin=-1,vmax=1,center=0, 
                cmap='bwr',square=False,lw=2,annot=True,cbar=True)
    ax.set_title('Linear Correlation Matrix')



def acf_and_pacf_plots(df, pool):
    """
    Plot the auto-correlation and partial auto-correlation plots for the net energy of a specific pool.
    
    Args:
    df (pd.DataFrame): DataFrame containing the time series data
    pool (int): Pool number
    """

    fig, ax = plt.subplots(2, 1, figsize=(15, 5))

    # Auto-correlation plot
    plot_acf(df['net_energy'][df["pool"] == pool], lags=50, ax=ax[0])
    ax[0].set_title(f'Autocorrelation for Pool {pool}')
    ax[0].set_ylabel('Autocorrelation', fontsize=12)

    # Partial auto-correlation plot
    plot_pacf(df['net_energy'][df["pool"] == pool], lags=50, ax=ax[1])
    ax[1].set_title(f'Partial Autocorrelation for Pool {pool}')
    ax[1].set_ylabel('Partial Autocorrelation', fontsize=12)
    ax[1].set_xlabel('Lagged Hours', fontsize=12)

    plt.tight_layout()

    plt.show()



def add_lag_columns(df, lag):
    """
    Add lag columns to the DataFrame.

    Args:
    df (pd.DataFrame): DataFrame containing the time series data
    lag (int): Number of lag columns to add

    Returns:
    new_df (pd.DataFrame): DataFrame containing the time series data with lag columns
    """

    new_df = df.copy()

    for i in range(1, lag + 1):
        column_name = f"net_energy_lag_{i}"                     # Create the column name
        new_df[column_name] = new_df["net_energy"].shift(i)     # Shift the net_energy column by i hours

    return new_df



def add_lag_columns_by_pool(df, lag):
    """
    Add lag columns to the DataFrame by pool.
    
    Args:
    df (pd.DataFrame): DataFrame containing the time series data
    lag (int): Number of lag columns to add
    
    Returns:
    result_df (pd.DataFrame): DataFrame containing the time series data with lag columns by pool
    """
    
    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame()
    
    # Iterate over each pool
    for pool in df["pool"].unique():
        pool_df = df[df["pool"] == pool].copy()                     # Filter the DataFrame for the current pool
        for i in range(1, lag + 1):
            column_name = f"net_energy_lag_{i}"                     # Create the column name    
            pool_df[column_name] = pool_df["net_energy"].shift(i)   # Shift the net_energy column by i hours
        result_df = pd.concat([result_df, pool_df])                 # Concatenate the results
    
    result_df.dropna(inplace=True)                                  # Drop rows with missing values due to the shift
    result_df.drop("date", axis=1, inplace=True)                    # Drop the date column

    return result_df.reset_index(drop=True) 



def train_xgb_forecast_model(df, model):
    """
    Train the XGBoost model on the net energy data.

    Args:
    df (pd.DataFrame): DataFrame containing the time series data
    model (XGBRegressor): XGBoost model
    """ 

    # Split the data into train and test according to case study requirements
    train = df.loc[:,df.columns != 'datetime_utc'][df['datetime_utc'] < '2023-01-01']
    test = df.loc[:,df.columns != 'datetime_utc'][df['datetime_utc'] >= '2023-01-01']

    # Split train into X and y
    xgb_X_train = train.drop('net_energy', axis=1)
    xgb_y_train = train['net_energy']

    # Split test into X and y
    xgb_X_test = test.drop('net_energy', axis=1)
    xgb_y_test = test['net_energy']

    # Fit the model
    model.fit(  X = xgb_X_train, 
                y = xgb_y_train, 
                eval_set = [(xgb_X_train, xgb_y_train), (xgb_X_test, xgb_y_test)], 
                verbose=True 
            )



def plot_pool_predictions(df, model, pool=0, zoomed=False):
    """
    Plot the predictions of the model for a given pool.

    Args:
    df (pd.DataFrame): DataFrame containing the time series data
    model (XGBRegressor): XGBoost model
    pool (int): Pool number
    zoomed (bool): True if the plot should be zoomed, False otherwise
    """

    # Filter the data for the selected pool
    pool_data = df[df[f"pool_{pool}"] == 1].copy()

    # Time Periods
    train_range = pool_data[pool_data["datetime_utc"] < "2023-01-01"]
    test_range = pool_data[pool_data["datetime_utc"] >= "2023-01-01"]

    # Explantory variables
    X_train = train_range.drop(['datetime_utc', 'net_energy'], axis=1)
    X_test = test_range.drop(['datetime_utc', 'net_energy'], axis=1)

    # Target variable
    y_train = train_range['net_energy']
    y_test = test_range['net_energy']

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # MAE
    test_mae = mean_absolute_error(y_test, test_pred)


    plt.figure(figsize=(15, 4))

    if zoomed:
        # Training and Test Data
        plt.plot(test_range["datetime_utc"][:100], y_test[:100], label='Prediction', color=pool_color[pool],  lw=0.5)

        # Fitted Values and Predictions
        plt.plot(test_range["datetime_utc"][:100], test_pred[:100], label='Predictions (Test)', color='black',  lw=0.5)
    else:
        # Training and Test Data
        plt.plot(train_range["datetime_utc"], y_train, label='Truth', color=pool_color[pool], lw=0.5)
        plt.plot(test_range["datetime_utc"], y_test, label='Prediction', color=pool_color[pool],  lw=0.5)

        # Fitted Values and Predictions
        plt.plot(train_range["datetime_utc"], train_pred, label='Fitted (Train)', color='black', lw=0.5)
        plt.plot(test_range["datetime_utc"], test_pred, label='Predictions (Test)', color='black',  lw=0.5)

    # Prediction Intervals
    plt.axvline(x=test_range["datetime_utc"].iloc[0], color='black', lw=0.5, linestyle='--', label='Prediction Start')

    # Add labels and legend
    handles = [plt.Line2D([0], [0], color=pool_color[pool], lw=2.5, label='Truth'),
                plt.Line2D([0], [0], color="black", lw=2.5, label='Prediction'),
                plt.Line2D([0], [0], color='black', lw=0.5, linestyle='--', label='Prediction Start')]
    
    plt.legend(handles=handles)
    if zoomed:
        plt.title(f'Net energy predictions for pool {pool} (MAE: {test_mae:.2f}) - Zoomed')
    else:
        plt.title(f'Net energy predictions for pool {pool} (MAE: {test_mae:.2f})')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Net Energy', fontsize=12)

    
    plt.show()


def plot_all_mae(df, model, pool_list):
    """
    Plot the Mean Absolute Error for each pool.

    Args:
    df (pd.DataFrame): DataFrame containing the time series data
    model (XGBRegressor): XGBoost model
    pool_list (list): List of pool numbers
    """
    
    test_maes = []

    for pool in pool_list:
        pool_data = df[df[f"pool_{pool}"] == 1].copy()

        # Time Periods
        train_range = pool_data[pool_data["datetime_utc"] < "2023-01-01"]
        test_range = pool_data[pool_data["datetime_utc"] >= "2023-01-01"]

        # Explanatory variables
        X_train = train_range.drop(['datetime_utc', 'net_energy'], axis=1)
        X_test = test_range.drop(['datetime_utc', 'net_energy'], axis=1)

        # Target variable
        y_test = test_range['net_energy']

        # Predictions
        test_pred = model.predict(X_test)

        # MAE
        test_maes.append(mean_absolute_error(y_test, test_pred))

    plt.figure(figsize=(15,3))
    
    # Plot the MAE for each pool
    plt.bar(pool_list, test_maes, color=pool_color)

    # Add labels and title
    plt.xticks(pool_list)
    plt.xlabel("Pool", fontsize=12)
    plt.ylabel("Mean Absolute Error", fontsize=12)
    plt.title(f"Mean Absolute Error by Pool (Mean: {np.mean(test_maes):.2f}, Median: {np.median(test_maes):.2f})", fontsize=14)
    
    plt.show()





# ---------- Task 2 Functions ---------- #

def remove_and_interpolate(df, column, threshold):
    """
    Identify values above a specified threshold, remove them, and linearly interpolate the missing values.
    
    Args:
    df (pd.DataFrame): DataFrame containing the data
    column (str): The column to check for threshold
    threshold (float): The threshold value

    Returns:
    df_interpolated (pd.DataFrame): DataFrame with missing values interpolated
    """

    # Identify values above the threshold
    df_filtered = df.copy()
    df_filtered.loc[df_filtered[column] > threshold, column] = None
    
    # Interpolate missing values
    df_interpolated = df_filtered.interpolate()
    
    return df_interpolated


def plot_electricity_price(df):
    """
    Plot the electricity prices over time.
    
    Args:
    df (pd.DataFrame): DataFrame containing the electricity prices
    """

    # Plotting dimensions
    plt.figure(figsize=(15, 5))
    
    # Plot the electricity prices
    plt.plot(df["datetime_utc"], df["euros_per_mwh"], color="#f08721")

    # Add labels and title
    plt.title("Electricity prices", fontsize=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (€)", fontsize=12)

    plt.tight_layout()
    plt.show()


def summarise_profits(daily_profits):
    """
    Print a summary of the daily profits.

    Args:
    daily_profits (list): List of daily profits
    """
    
    print(f"Total profit over the period: {sum(daily_profits):.2f}€")
    print(f"Average daily profit: {np.mean(daily_profits):.2f}€")
    print(f"-------------------------------------")
    print(f"Highest daily profit: {max(daily_profits):.2f}€")
    print(f"Lowest daily profit: {min(daily_profits):.2f}€")



def plot_hourly_profit(electricity, hourly_trading):
    """
    Plot the hourly profit across the whole period.

    Args:
    electricity (pd.DataFrame): DataFrame containing the electricity prices
    hourly_trading (pd.DataFrame): DataFrame containing the hourly trading data
    """
    
    # Plot hourly profit across the whole period
    plt.figure(figsize=(15, 7))

    plt.subplot(2, 1, 1)

    # Plot hourly profit
    plt.plot(hourly_trading["datetime_utc"], hourly_trading["profit"], color='blue')

    # Add labels and title
    plt.xlabel('Hours', fontsize=12)
    plt.ylabel('Profit (€)', fontsize=12)
    plt.title('Hourly Profit')

    # Plotting Style
    plt.grid(True)
    plt.tight_layout()

    plt.subplot(2, 1, 2)

    # Plot hourly profit
    plt.plot(electricity.index, np.cumsum(hourly_trading["profit"]), color='blue')

    # Add labels and title
    plt.xlabel('Hours', fontsize=12)
    plt.ylabel('Cumulative Profit (€)', fontsize=12)
    plt.title('Cumulative Profit')

    # Plotting Style
    plt.grid(True)
    plt.tight_layout()


    plt.show()



def plot_hourly_decisions(hourly_trading):
    """
    Plot the average hourly buying and selling decisions.

    Args:
    hourly_trading (pd.DataFrame): DataFrame containing the hourly trading data
    """

    # Assuming your DataFrame is named `hourly_trading`
    hourly_trading['hour'] = hourly_trading['datetime_utc'].dt.hour

    fig, ax1 = plt.subplots(figsize=(15, 5))

    # Average hourly charging decisions
    hourly_buy_means = hourly_trading.groupby('hour')['charging_decision'].mean().reset_index()
    ax1.plot(hourly_buy_means['hour'], hourly_buy_means['charging_decision'], '-x', color='green', label='Buying', lw=1)

    # Average hourly discharging decisions
    hourly_sell_means = hourly_trading.groupby('hour')['discharging_decision'].mean().reset_index()
    ax1.plot(hourly_sell_means['hour'], hourly_sell_means['discharging_decision'], '-x', color='red', label='Selling', lw=1)

    # Decision threshold
    ax1.axhline(y=0.5, color='black', linestyle='--',  alpha=0.5)

    # Create a secondary y-axis
    ax2 = ax1.twinx()

    # Average hourly electricity prices
    hourly_price_means = hourly_trading.groupby('hour')['price'].mean().reset_index()
    ax2.plot(hourly_price_means['hour'], hourly_price_means['price'], '-x', color="#f08721", label='Price', lw=1)

    # Add labels and title
    ax1.set_xlabel('Hour of the Day', fontsize=12)
    ax1.set_xticks(range(24))
    ax1.set_ylabel('Electricity (MW)', fontsize=12)
    ax2.set_ylabel('Price (€)', fontsize=12)
    plt.title('Average Price, Buying and Selling Decisions by Hour of the Day')

    # Add legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=12)

    # Add grid and layout adjustments
    fig.tight_layout()

    plt.show()



def plot_hourly_state_of_charge(hourly_trading, hourly_s):
    """
    Plot the averaged hourly battery's state of charge and hourly state of charge for the first day.

    Args:
    hourly_trading (pd.DataFrame): DataFrame containing the hourly trading data
    hourly_s (list): List of hourly state of charge
    """

    plt.figure(figsize=(15, 5))

    # Group by hour and calculate the average state of charge
    hourly_state_means = hourly_trading.groupby('hour')["state_of_charge"].mean().reset_index()

    # Plot the battery's state of charge
    plt.plot(hourly_state_means['hour'], hourly_state_means["state_of_charge"], '-x', color='purple', label='Average', lw=1)
    plt.plot(hourly_state_means['hour'], hourly_s[0], '-o', color='black', lw=1, label='First Day')

    # Add labels and title
    plt.xlabel('Hours of the Day', fontsize=12)
    plt.xticks(np.arange(24))
    plt.ylabel('State of Charge (MWh)', fontsize=12)
    plt.title("Battery's State of Charge")
    plt.grid(True)
    plt.legend()

    plt.show()



def comparing_profits_for_different_battery_capacities(electricity, hourly_trading, hourly_trading_2):
    """
    Compare the daily profits for two different battery capacities.

    Args:
    electricity (pd.DataFrame): DataFrame containing the electricity prices
    hourly_trading (pd.DataFrame): DataFrame containing the hourly trading data for the 1MWh battery capacity
    hourly_trading_2 (pd.DataFrame): DataFrame containing the hourly trading data for the 2MWh battery capacity
    """

    # Plotting
    plt.figure(figsize=(15, 6))

    # Plot hourly profit
    plt.plot(electricity.index, np.cumsum(hourly_trading["profit"]), color='blue', label='Battery Capacity 1MWh')
    plt.plot(electricity.index, np.cumsum(hourly_trading_2["profit"]), color='lightblue', label='Battery Capacity 2MWh')
    
    # Add labels and title
    plt.xlabel('Hours', fontsize=12)
    plt.ylabel('Cumulative Profit (€)', fontsize=12)
    plt.title('Cumulative Profit')

    # Plotting Style
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()



def sample_price_trajectories(n_trajectory, model, n_samples=24, length=24):
    """
    Sample price trajectories from the Gaussian Process.

    Args:
    n_trajectory (int): The number of price trajectories to sample
    model (GaussianProcessRegressor): The Gaussian Process regressor
    n_samples (int): The number of time points to sample
    length (int): Length of the time series

    Returns:
    pd.DataFrame: DataFrame containing the sampled price trajectories
    """
    
    x_samples = np.linspace(0, length-1, n_samples).reshape(-1, 1)
    sample_paths = model.sample_y(x_samples, n_samples=n_trajectory, random_state=7)

    return pd.DataFrame(sample_paths)



def plot_day_price_trajectories(true_price, gp):
    """
    Plot the observed price trajectories and the Gaussian Process regression.

    Args:
    true_price (pd.DataFrame): DataFrame containing the true price trajectories
    gp (GaussianProcessRegressor): The Gaussian Process regressor
    """

    # Prediction
    x_pred = np.linspace(0, 23, 1000).reshape(-1, 1)
    x_samples = np.linspace(0, 23, 24).reshape(-1, 1)
    y_pred, sigma = gp.predict(x_pred, return_std=True)

    # True prices
    plt.figure(figsize=(15, 5))
    for i in range(24):
        plt.scatter(i, true_price.iloc[i], color='#f08721', label='True Price' if i == 0 else None)
    
    # GP regression
    plt.fill_between(x_pred.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, color='#f08721', alpha=0.2, label='95% Confidence Interval')
    
    # Drawing multiple sample paths
    sample_paths = sample_price_trajectories(n_trajectory=5, model = gp, n_samples=24, length=24)
    for i in range(sample_paths.shape[1]):
        plt.plot(x_samples, sample_paths.iloc[:, i], '-x', lw=0.8, color='#f08721', label='Sample Trajectory' if i == 0 else None, alpha=0.5)

    # Add labels and title
    plt.xlabel('Hours of the Day', fontsize=12)
    plt.xticks(np.arange(24))
    plt.ylabel('Price (€)', fontsize=12)
    plt.legend()
    plt.title('Gaussian Process Regression on Electricity Prices')

    plt.tight_layout()
    plt.show()



def smooth_dataframe(df):
    """
    Smooths each column in a given DataFrame using a Gaussian filter.

    Parameters:
    df (pd.DataFrame): The input DataFrame with time series data.

    Returns:
    pd.DataFrame: The DataFrame with smoothed time series.
    """

    seed = 3
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate random sigmas for price trajectories
    sigmas = np.random.uniform(0.3, 0.7, df.shape[1])

    # Smooth each trajectory using a Gaussian filter
    smoothed_df = pd.DataFrame({col: gaussian_filter1d(df[col], sigma=sigmas[i]) for i, col in enumerate(df.columns)})

    return smoothed_df



def plot_day_price_smoothed_trajectories(y, price_samples_smoothed):
    """
    Plot the true and sampled price trajectories for a day.
    
    Args:
    y (pd.Series): The true price trajectories for a day.
    price_samples_smoothed (pd.DataFrame): The sampled price trajectories for a day.
    """

    plt.figure(figsize=(15, 5))

    # True prices for a day
    plt.plot(np.arange(24), y[:24], color="#f5ab64", label="True prices")

    # Sampled price trajectories for a day
    for i in price_samples_smoothed.columns:
        plt.plot(np.arange(24), price_samples_smoothed[i][:24], '-x', color='#f5ab64', alpha=0.5, lw=0.5, label="Sampled prices" if i == 0 else "")

    # Add labels and title
    plt.title("True and Sampled Price Trajectories for a Day")
    plt.xlabel('Hours of the Day', fontsize=12)
    plt.xticks(np.arange(24))
    plt.ylabel("Price (€)", fontsize=12)
    plt.legend()

    plt.show()



def plot_kde_density(mean, lp_mean, total, lp_total):
    """
    Plot the SAA kernel density for the mean and total profits.

    Args:
    mean (list): List of mean daily profits
    lp_mean (float): LP mean daily profit
    total (list): List of total profits
    lp_total (float): LP total profit
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    sns.kdeplot(mean, fill=True, color= "#f08721", label='SAA Mean Daily Profits')
    plt.axvline(lp_mean, color='#f08721', linestyle='dashed', linewidth=1, label='LP Mean Daily Profit')
    plt.title('Mean Daily Profits Density')
    plt.xlabel('Mean Daily Profits (€)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.kdeplot(total, fill=True, color= "#f08721", label='SAA Total Profits')
    plt.axvline(lp_total, color='#f08721', linestyle='dashed', linewidth=1, label='LP Total Profit')
    plt.title('Total Profits Density')
    plt.xlabel('Total Profits (€)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.show()