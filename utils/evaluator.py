import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mase(y_true, y_pred, y_train_raw, seasonality=24):
    # Ensure training data is a 1D flat array for differencing
    y_train = y_train_raw.flatten()
    
    # Calculate the MAE of the seasonal naive forecast on the training set
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)
    
    # Avoid division by zero
    if scale == 0: return np.nan
    
    # Return the ratio of model error to naive error
    return np.mean(np.abs(y_true - y_pred)) / scale
