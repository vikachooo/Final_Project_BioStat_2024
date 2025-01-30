import pandas as pd
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import warnings
import itertools

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import het_breuschpagan
from prophet import Prophet

### ARIMA Functions
def train_arima_model(data, order, train_month=84, seasonal_order=None, exog=None, trend=None):
    """
    Function to train ARIMA/ARIMAX/SARIMAX models.
    
    Parameters:
        data (pd.Series): Time series data (index should be datetime-like).
        order (tuple): ARIMA order (p, d, q).
        seasonal_order (tuple, optional): Seasonal order (P, D, Q, s).
        exog (pd.DataFrame, optional): External regressors for ARIMAX.
        train_split (float or int): Fraction or number of data points for training.
        trend (str, optional): Trend component ('n', 'c', 't', 'ct').
    
    Returns:
        dict: A dictionary containing the model, predictions, and performance metrics.
    """
    # Split the data into training and testing sets 
    train_data = data.iloc[:train_month]
    test_data = data.iloc[train_month:]

    train_size = train_data.shape[0]
    
    if exog is not None:
        exog_train = np.array(exog.iloc[:train_size])
        exog_test = np.array(exog.iloc[train_size:])
    else:
        exog_train, exog_test = None, None
    
    # Define the model
    model = SARIMAX(train_data,
                    order=order,
                    seasonal_order=seasonal_order if seasonal_order else (0, 0, 0, 0),
                    exog=exog_train,
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    
    # Fit the model
    model = model.fit()
    
    # In-sample predictions
    train_predictions = model.fittedvalues
    
    return model

def make_metrics_dataframe(model, data_model_train, data_model_test, main_column, exog_test=None, plotting=True, *args, **kwargs):
    # Create a DataFrame for future predictions (extend the time range if necessary)
    forecast_steps = data_model_test.shape[0]  # Number of periods to forecast
    forecast = model.get_forecast(steps=forecast_steps, exog=exog_test)
    forecast_values = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # Make final dataset
    train_size = data_model_train.shape[0]
    test_size = data_model_test.shape[0]
    
    data_check_metrics = pd.concat([data_model_train, data_model_test])
    data_check_metrics = data_check_metrics[['Date', 'Date_Graph', main_column]]
    data_check_metrics.loc[:, f"{main_column}_Fitted"] = model.fittedvalues
    data_check_metrics.loc[:, f"{main_column}_Prediction"] = pd.concat([pd.Series([None]*train_size), forecast_values])
    data_check_metrics.loc[:, f"{main_column}_Prediction_CI_low"] = pd.concat([pd.Series([None]*train_size), forecast_ci[f'lower {main_column}']])
    data_check_metrics.loc[:, f"{main_column}_Prediction_CI_upp"] = pd.concat([pd.Series([None]*train_size), forecast_ci[f'upper {main_column}']])
    
    data_check_metrics = data_check_metrics.loc[data_check_metrics.Date <= pd.to_datetime("2024-10-10")]
    return data_check_metrics

def calculate_model_metrics(model, data_check_metrics, main_column):
    metrics = []  # List to store metrics for TRAIN and TEST

    # Calculate metrics - TRAIN
    mae_train = np.mean(np.abs(data_check_metrics[main_column] - data_check_metrics[f"{main_column}_Fitted"]))
    mse_train = np.mean((data_check_metrics[main_column] - data_check_metrics[f"{main_column}_Fitted"]) ** 2)
    rmse_train = np.sqrt(mse_train)
    mape_train = np.mean(np.abs((data_check_metrics[main_column] - data_check_metrics[f"{main_column}_Fitted"]) / data_check_metrics[main_column])) * 100
    aic_train = model.aic

    # Store TRAIN metrics in the list
    metrics.append({
        "Dataset": "TRAIN",
        "MAE": mae_train,
        "MSE": mse_train,
        "RMSE": rmse_train,
        "MAPE (%)": mape_train,
        "AIC": aic_train
    })

    # Calculate metrics - TEST
    mae_test = np.mean(np.abs(data_check_metrics[main_column] - data_check_metrics[f"{main_column}_Prediction"]))
    mse_test = np.mean((data_check_metrics[main_column] - data_check_metrics[f"{main_column}_Prediction"]) ** 2)
    rmse_test = np.sqrt(mse_test)
    mape_test = np.mean(np.abs((data_check_metrics[main_column] - data_check_metrics[f"{main_column}_Prediction"]) / data_check_metrics[main_column])) * 100

    # Store TEST metrics in the list
    metrics.append({
        "Dataset": "TEST",
        "MAE": mae_test,
        "MSE": mse_test,
        "RMSE": rmse_test,
        "MAPE (%)": mape_test,
        "AIC": None
    })

    # Convert metrics to a DataFrame
    metrics_df = pd.DataFrame(metrics)
    metrics_df.index = metrics_df['Dataset']
    metrics_df.drop(columns=['Dataset'], inplace=True)
    metrics_df = metrics_df.T

    # Print the metrics
    print("*** TRAIN ***")
    print(f"MAE: {mae_train:.2f}")
    print(f"MSE: {mse_train:.2f}")
    print(f"RMSE: {rmse_train:.2f}")
    print(f"MAPE: {mape_train:.2f}%")
    print(f"AIC: {aic_train}")

    print("\n*** TEST ***")
    print(f"MAE: {mae_test:.2f}")
    print(f"MSE: {mse_test:.2f}")
    print(f"RMSE: {rmse_test:.2f}")
    print(f"MAPE: {mape_test:.2f}%")

    return metrics_df    

def model_performance(data_check_metrics, main_column):
    # Assume `df` is your DataFrame with columns 'real' and 'fitted'
    train_size = data_check_metrics[f"{main_column}_Fitted"].dropna().shape[0]
    # Replace 'real' and 'fitted' with your actual column names
    real = data_check_metrics[main_column].iloc[0:train_size]
    fitted = data_check_metrics[f"{main_column}_Fitted"].iloc[0:train_size]
    
    # Calculate residuals
    data_check_metrics['residuals'] = real - fitted
    residuals = data_check_metrics['residuals'].dropna()

    from statsmodels.stats.diagnostic import acorr_ljungbox

    # Perform the Ljung-Box test on residuals
    # df['residuals'] should already be computed
    ljung_box_results = acorr_ljungbox(data_check_metrics['residuals'].dropna(), lags=[10], return_df=True)
    
    # Display results
    print('\n')
    print('Ljung-Box Test')
    print(ljung_box_results)

    # test_stat, p_value, _, _ = het_breuschpagan(residuals, exog_het=np.arange(len(residuals)))
    # # Display results
    # print('\n')
    # print('Breusch-Pagan Test')
    # print(test_stat, p_value)

    # Residuals plot
    plt.figure(figsize=(10, 6))
    plt.plot(residuals, label='Residuals', color='blue')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title('Residuals Over Time')
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.legend()
    plt.show()
    
    # Histogram of residuals
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, color='gray', edgecolor='black')
    plt.title('Histogram of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.show()
    
    # ACF and PACF of residuals
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_acf(residuals, lags=10, ax=plt.gca(), title="ACF of Residuals")
    plt.subplot(1, 2, 2)
    plot_pacf(residuals, lags=10, ax=plt.gca(), title="PACF of Residuals")
    plt.tight_layout()
    plt.show()

def rolling_prediction_function(data_used_model, main_column, arima_order, seasonal_order=None, train_month_start=84, train_month_end=94, exog=None):
    data_used_train = data_used_model.iloc[:train_month_start]
    data_used_test = data_used_model.iloc[train_month_start:]
    
    rolling_prediction = []
    rolling_prediction_ci_low = []
    rolling_prediction_ci_upp = []
    
    model_first = train_arima_model(data_used_model[main_column], arima_order, train_month=train_month_start, exog=exog, seasonal_order=seasonal_order)
    if exog is not None:
        first_data_check_metrics = make_metrics_dataframe(model_first, data_used_train, data_used_test, main_column, exog_test=exog.loc[data_used_test.index])
    else:
        first_data_check_metrics = make_metrics_dataframe(model_first, data_used_train, data_used_test, main_column)
    
    for train_month in range(train_month_start, train_month_end):
        #print(train_month)
        model = train_arima_model(data_used_model[main_column], arima_order, train_month=train_month, seasonal_order=seasonal_order, exog=exog)
        if exog is not None:
            data_check_metrics = make_metrics_dataframe(model, data_used_train, data_used_test, main_column, exog_test=np.array(exog.iloc[train_month_start:]))
        else:
            data_check_metrics = make_metrics_dataframe(model, data_used_train, data_used_test, main_column, exog_test=None)
        rolling_prediction.append(data_check_metrics[f"{main_column}_Prediction"].dropna().iloc[0])
        rolling_prediction_ci_low.append(data_check_metrics[f"{main_column}_Prediction_CI_low"].dropna().iloc[0])
        rolling_prediction_ci_upp.append(data_check_metrics[f"{main_column}_Prediction_CI_upp"].dropna().iloc[0])
    
    final_data_check_metrics = first_data_check_metrics.copy()
    final_data_check_metrics[f"{main_column}_Prediction"] = pd.concat([pd.Series([None]*train_month_start), pd.Series(rolling_prediction)], ignore_index=True)
    final_data_check_metrics[f"{main_column}_Prediction_CI_low"] = pd.concat([pd.Series([None]*train_month_start), pd.Series(rolling_prediction_ci_low)], ignore_index=True)
    final_data_check_metrics[f"{main_column}_Prediction_CI_upp"] = pd.concat([pd.Series([None]*train_month_start), pd.Series(rolling_prediction_ci_upp)], ignore_index=True)

    return final_data_check_metrics, model_first

def grid_search_arima(data, min_p, max_p, min_q, max_q, d, 
                      min_P, max_P, min_Q, max_Q, 
                      min_D, max_D, seasonal_period, exog=None,
                     trends=None):
    """
    Perform a grid search to find the best ARIMA/SARIMA model.
    
    Parameters:
        data (pd.Series): Time series data (index should be datetime-like).
        min_p (int): Minimum value for p (AR order).
        max_p (int): Maximum value for p (AR order).
        min_q (int): Minimum value for q (MA order).
        max_q (int): Maximum value for q (MA order).
        d (int): Differencing order.
        min_P (int): Minimum value for seasonal P (seasonal AR order).
        max_P (int): Maximum value for seasonal P (seasonal AR order).
        min_Q (int): Minimum value for seasonal Q (seasonal MA order).
        max_Q (int): Maximum value for seasonal Q (seasonal MA order).
        min_D (int): Minimum value for seasonal D (seasonal differencing).
        max_D (int): Maximum value for seasonal D (seasonal differencing).
        seasonal_period (int): Seasonal period (s).
        exog (pd.DataFrame, optional): External regressors for ARIMAX.
        
    Returns:
        pd.DataFrame: Grid search results with model orders and AIC values.
    """
    results = []
    if not trends:
        trends = ['n', 'c', 't', 'ct']  # Trend options
    
    # Generate parameter grid
    for p, q, P, Q, D, trend in itertools.product(
        range(min_p, max_p + 1),
        range(min_q, max_q + 1),
        range(min_P, max_P + 1),
        range(min_Q, max_Q + 1),
        range(min_D, max_D + 1),
        trends
    ):
        try:
            # Fit SARIMAX model
            model = train_arima_model(data, order=(p, d, q), seasonal_order=(P, D, Q, seasonal_period), trend=trend,
                                     exog=exog)
            
            # Store results
            results.append({
                'p': p,
                'q': q,
                'd': d,
                'P': P,
                'Q': Q,
                'D': D,
                'seasonal_period': seasonal_period,
                'trend': trend,
                'AIC': 9999 if model.aic < 500 else model.aic
            })
        except Exception as e:
            print(p,q,d,P,Q,D)
            print('error')
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def plot_model_results(data, main_column, date_column='Date_Graph', 
                         prediction_column=None, fitted_column=None,
                         ci_low_column=None, ci_upp_column=None, 
                         cutoff_date=None, vertical_line_date=None, 
                         vertical_line_ymin=None, vertical_line_ymax=None,
                      figsize=None):
    """
    Plot ARIMA model results, including actual, fitted, and predicted values with confidence intervals.

    Parameters:
        data (pd.DataFrame): DataFrame containing the time series data.
        main_column (str): Column name for the actual values.
        date_column (str): Column name for the date. Default is 'Date_Graph'.
        prediction_column (str): Column name for predictions.
        fitted_column (str): Column name for fitted values.
        ci_low_column (str): Column name for lower confidence interval.
        ci_upp_column (str): Column name for upper confidence interval.
        cutoff_date (str): Date to filter the data for predictions. Format: 'YYYY-MM-DD'.
        vertical_line_date (str): Date for a vertical line. Format: 'YYYY-MM-DD'.
        vertical_line_ymin (float): Y-axis minimum for the vertical line.
        vertical_line_ymax (float): Y-axis maximum for the vertical line.

    Returns:
        None: Displays the plot.
    """
    if figsize is None:
        figsize = (10, 6)
    # Ensure date_column is in datetime format
    data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

    # Plot the data
    plt.figure(figsize=figsize)
    plt.plot(data[date_column], data[main_column], label='Actual', color="orange", alpha=0.7)

    if fitted_column:
        plt.plot(data[date_column], data[fitted_column], label='Fitted', color='blue', linestyle="dashed", linewidth=3)

    if prediction_column:
        plt.plot(data[date_column], data[prediction_column], label='Predicted', linestyle="dashdot", linewidth=3, color='red')

    if vertical_line_date:
        plt.vlines(x=pd.to_datetime(vertical_line_date), linestyles='--', color='orange', ymin=vertical_line_ymin, ymax=vertical_line_ymax)

    # Add titles and labels
    plt.title('Model - Actual vs Fitted and Forecast')
    plt.xlabel('Date')
    plt.ylabel('Load')
    plt.ylim(data[main_column].min() - 200, data[main_column].max() + 200)
    plt.legend()
    plt.grid()

    # Xticks
    xticks = data[date_column][::2]  # Select every second tick
    plt.xticks(xticks, xticks.dt.strftime('%Y-%m-%d'), rotation=45)
    plt.show()

    if cutoff_date:
        # Filter the dataset for plotting predictions
        filtered_data = data.loc[data[date_column] > pd.to_datetime(cutoff_date)]
        
        if prediction_column:
            filtered_data[prediction_column] = filtered_data[prediction_column].fillna(filtered_data[fitted_column])

        # Plot the filtered data
        plt.figure(figsize=figsize)
        plt.plot(filtered_data[date_column], filtered_data[main_column], label='Actual', color="orange", alpha=0.7)

        if prediction_column:
            plt.plot(filtered_data[date_column], filtered_data[prediction_column], label='Predicted', linestyle="dashdot", linewidth=3, color='red')

        if ci_low_column and ci_upp_column:
            filtered_data.loc[filtered_data["Date"] == pd.to_datetime("2023-12-01"), 
                f"{main_column}_Prediction"] = data.loc[data["Date"] == pd.to_datetime("2023-12-01"), f"{main_column}_Fitted"]
            filtered_data = filtered_data.fillna(1500)
            #print(filtered_data.head())
            plt.fill_between(filtered_data[date_column], 
                             filtered_data[ci_low_column], 
                             filtered_data[ci_upp_column], 
                             color='pink', alpha=0.3, label='Confidence Interval')

        # Add titles and labels
        plt.title('Model - Actual vs Forecast (Filtered)')
        plt.xlabel('Date')
        plt.ylabel('Load')
        if ci_low_column and ci_upp_column:
            plt.ylim(filtered_data[ci_low_column].min() - 100, filtered_data[ci_upp_column].max() + 100)
        else:
            plt.ylim(filtered_data[main_column].min() - 200, filtered_data[main_column].max() + 200)
        plt.legend()
        plt.grid()

        # Xticks
        xticks = filtered_data[date_column][::2]
        plt.xticks(xticks, xticks.dt.strftime('%Y-%m-%d'), rotation=45)
        plt.show()

### Prophet Functions
def calculate_aic(col_true, col_prediction, num_params=50):
    # To common indexes
    indexes = set(col_true.dropna().index) & set(col_prediction.dropna().index)
    col_true = col_true[indexes]
    col_prediction = col_prediction[indexes]

    # Calculate residuals and variance
    residuals = col_true.values - col_prediction.values
    rss = np.sum(residuals ** 2)
    sigma2 = np.var(residuals)

    # Number of parameters
    k = num_params
    
    # Number of observations
    n = len(col_true.values)

    #Likelihood
    log_likelihood = -n / 2 * (np.log(2 * np.pi) + np.log(sigma2) + rss / (n * sigma2))

    # AIC
    aic = 2 * k - 2 * log_likelihood

    return aic

def prophet_model_train(model, data_model_train, data_model_test, main_column, plotting=True, *args, **kwargs):

    # Initialize and fit the Prophet model
    model.fit(data_model_train)
    
    # Create a DataFrame for future predictions (extend the time range if necessary)
    future = model.make_future_dataframe(periods=data_model_test.shape[0], freq='M')  # Forecast 10 months into the future
    forecast = model.predict(future)

    if plotting:      
        # Plot forecast components (trend, seasonality, etc.)
        fig = model.plot_components(forecast, figsize=(6, 6))
        plt.xticks(rotation=45)
        plt.show()

    # Calculate Metrics
    data_check_metrics = pd.concat([data_model_train, data_model_test])
    data_check_metrics['Date'] = data_check_metrics['ds']
    data_check_metrics[main_column] = data_check_metrics['y']
    data_check_metrics = data_check_metrics[['Date', 'Date_Graph', main_column]]
    data_check_metrics = data_check_metrics.reset_index()
    train_size = data_model_train.shape[0]
    test_size = data_model_test.shape[0]

    # Making dataset to calculate metrics
    fitted = forecast.iloc[0:train_size]['yhat']
    data_check_metrics[f"{main_column}_Fitted"] = pd.concat([fitted, pd.Series([None]*test_size)]).values
    
    # Add Predictions and Confidence Intervals
    predicted = forecast.iloc[train_size:data_check_metrics.shape[0]]['yhat']
    data_check_metrics[f"{main_column}_Prediction"] = pd.concat([pd.Series([None]*train_size), predicted]).values
    
    data_check_metrics[f"{main_column}_Prediction_CI_low"] = forecast['yhat_lower'].values
        
    data_check_metrics[f"{main_column}_Prediction_CI_upp"] = forecast['yhat_upper'].values
    
    return model, data_check_metrics, forecast

def calculate_model_metrics_prophet(data_check_metrics, main_column):
    metrics = []  # List to store metrics for TRAIN and TEST

    # Calculate metrics - TRAIN
    mae_train = np.mean(np.abs(data_check_metrics[main_column] - data_check_metrics[f"{main_column}_Fitted"]))
    mse_train = np.mean((data_check_metrics[main_column] - data_check_metrics[f"{main_column}_Fitted"]) ** 2)
    rmse_train = np.sqrt(mse_train)
    mape_train = np.mean(np.abs((data_check_metrics[main_column] - data_check_metrics[f"{main_column}_Fitted"]) / data_check_metrics[main_column])) * 100
    aic_train = calculate_aic(data_check_metrics[main_column], data_check_metrics[f"{main_column}_Fitted"])
    
    # Store TRAIN metrics in the list
    metrics.append({
        "Dataset": "TRAIN",
        "MAE": mae_train,
        "MSE": mse_train,
        "RMSE": rmse_train,
        "MAPE (%)": mape_train,
        "AIC": aic_train
    })

    # Calculate metrics - TEST
    mae_test = np.mean(np.abs(data_check_metrics[main_column] - data_check_metrics[f"{main_column}_Prediction"]))
    mse_test = np.mean((data_check_metrics[main_column] - data_check_metrics[f"{main_column}_Prediction"]) ** 2)
    rmse_test = np.sqrt(mse_test)
    mape_test = np.mean(np.abs((data_check_metrics[main_column] - data_check_metrics[f"{main_column}_Prediction"]) / data_check_metrics[main_column])) * 100

    # Store TEST metrics in the list
    metrics.append({
        "Dataset": "TEST",
        "MAE": mae_test,
        "MSE": mse_test,
        "RMSE": rmse_test,
        "MAPE (%)": mape_test,
        "AIC": None
    })

    # Convert metrics to a DataFrame
    metrics_df = pd.DataFrame(metrics)
    metrics_df.index = metrics_df['Dataset']
    metrics_df.drop(columns=['Dataset'], inplace=True)
    metrics_df = metrics_df.T

    # Print the metrics
    print("*** TRAIN ***")
    print(f"MAE: {mae_train:.2f}")
    print(f"MSE: {mse_train:.2f}")
    print(f"RMSE: {rmse_train:.2f}")
    print(f"MAPE: {mape_train:.2f}%")
    print(f"AIC: {aic_train}")

    print("\n*** TEST ***")
    print(f"MAE: {mae_test:.2f}")
    print(f"MSE: {mse_test:.2f}")
    print(f"RMSE: {rmse_test:.2f}")
    print(f"MAPE: {mape_test:.2f}%")

    return metrics_df

def calcualate_model_performance_prophet(model, data_check_metrics, main_column):
    # Assume `df` is your DataFrame with columns 'real' and 'fitted'
    train_size = data_check_metrics[f"{main_column}_Fitted"].dropna().shape[0]
    # Replace 'real' and 'fitted' with your actual column names
    real = data_check_metrics[main_column].iloc[0:train_size]
    fitted = data_check_metrics[f"{main_column}_Fitted"].iloc[0:train_size]
    
    # Calculate residuals
    data_check_metrics['residuals'] = real - fitted
    residuals = data_check_metrics['residuals'].dropna()

    from statsmodels.stats.diagnostic import acorr_ljungbox

    # Perform the Ljung-Box test on residuals
    # df['residuals'] should already be computed
    ljung_box_results = acorr_ljungbox(data_check_metrics['residuals'].dropna(), lags=[10], return_df=True)
    
    # Display results
    print('\n')
    print('Ljung-Box Test')
    print(ljung_box_results)

    # Residuals plot
    plt.figure(figsize=(10, 6))
    plt.plot(residuals, label='Residuals', color='blue')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title('Residuals Over Time')
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.legend()
    plt.show()
    
    # Histogram of residuals
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, color='gray', edgecolor='black')
    plt.title('Histogram of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.show()
    
    # ACF and PACF of residuals
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_acf(residuals, lags=10, ax=plt.gca(), title="ACF of Residuals")
    plt.subplot(1, 2, 2)
    plot_pacf(residuals, lags=10, ax=plt.gca(), title="PACF of Residuals")
    plt.tight_layout()
    plt.show()

def plot_model_results_prophet(data, main_column, date_column='Date_Graph', 
                         prediction_column=None, fitted_column=None,
                         ci_low_column=None, ci_upp_column=None, 
                         cutoff_date=None, vertical_line_date=None, 
                         vertical_line_ymin=None, vertical_line_ymax=None,
                      figsize=None):
    """
    Plot Prophet model results, including actual, fitted, and predicted values with confidence intervals.

    Parameters:
        data (pd.DataFrame): DataFrame containing the time series data.
        main_column (str): Column name for the actual values.
        date_column (str): Column name for the date. Default is 'Date_Graph'.
        prediction_column (str): Column name for predictions.
        fitted_column (str): Column name for fitted values.
        ci_low_column (str): Column name for lower confidence interval.
        ci_upp_column (str): Column name for upper confidence interval.
        cutoff_date (str): Date to filter the data for predictions. Format: 'YYYY-MM-DD'.
        vertical_line_date (str): Date for a vertical line. Format: 'YYYY-MM-DD'.
        vertical_line_ymin (float): Y-axis minimum for the vertical line.
        vertical_line_ymax (float): Y-axis maximum for the vertical line.

    Returns:
        None: Displays the plot.
    """
    if figsize is None:
        figsize = (10, 6)
    # Ensure date_column is in datetime format
    data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

    # Plot the data
    plt.figure(figsize=figsize)
    plt.plot(data[date_column], data[main_column], label='Actual', color="orange", alpha=0.7)

    if fitted_column:
        plt.plot(data[date_column], data[fitted_column], label='Fitted', color='blue', linestyle="dashed", linewidth=3)

    if prediction_column:
        plt.plot(data[date_column], data[prediction_column], label='Predicted', linestyle="dashdot", linewidth=3, color='red')

    if vertical_line_date:
        plt.vlines(x=pd.to_datetime(vertical_line_date), linestyles='--', color='orange', ymin=vertical_line_ymin, ymax=vertical_line_ymax)

    # Add titles and labels
    plt.title('Model - Actual vs Fitted and Forecast')
    plt.xlabel('Date')
    plt.ylabel('Load')
    plt.ylim(data[main_column].min() - 200, data[main_column].max() + 200)
    plt.legend()
    plt.grid()

    # Xticks
    xticks = data[date_column][::2]  # Select every second tick
    plt.xticks(xticks, xticks.dt.strftime('%Y-%m-%d'), rotation=45)
    plt.show()

    if cutoff_date:
        # Filter the dataset for plotting predictions
        filtered_data = data.loc[data[date_column] > pd.to_datetime("2023-11-30")]
        
        if prediction_column:
            filtered_data[prediction_column] = filtered_data[prediction_column].fillna(filtered_data[fitted_column])

        # Plot the filtered data
        plt.figure(figsize=figsize)
        plt.plot(filtered_data[date_column], filtered_data[main_column], label='Actual', color="orange", alpha=0.7)

        if prediction_column:
            plt.plot(filtered_data[date_column], filtered_data[prediction_column], label='Predicted', linestyle="dashdot", linewidth=3, color='red')

        if ci_low_column and ci_upp_column:
            filtered_data.loc[filtered_data["Date"] == pd.to_datetime("2023-12-01"), 
                f"{main_column}_Prediction"] = data.loc[data["Date"] == pd.to_datetime("2023-12-01"), f"{main_column}_Fitted"]
            filtered_data = filtered_data.fillna(1500)
            #print(filtered_data.head())
            plt.fill_between(filtered_data[date_column], 
                             filtered_data[ci_low_column], 
                             filtered_data[ci_upp_column], 
                             color='pink', alpha=0.3, label='Confidence Interval')

        # Add titles and labels
        plt.title('Model - Actual vs Forecast (Filtered)')
        plt.xlabel('Date')
        plt.ylabel('Load')
        if ci_low_column and ci_upp_column:
            plt.ylim(filtered_data[ci_low_column].min() - 100, filtered_data[ci_upp_column].max() + 100)
        else:
            plt.ylim(filtered_data[main_column].min() - 200, filtered_data[main_column].max() + 200)
        plt.legend()
        plt.grid()

        # Xticks
        xticks = filtered_data[date_column][::2]
        plt.xticks(xticks, xticks.dt.strftime('%Y-%m-%d'), rotation=45)
        plt.show()