from statsmodels.tsa import ar_model
from statsmodels.tsa.arima.model import ARIMA
import scipy.stats as st
import numpy as np


def AR_1_phi_parameter_estimation(series):
    """This function takes an array of AR(1) time series as an argument
    and returns  a list of the phi1 estimated parameter for each series"""
    ar_list = []
    for serie in series:
        tsmodel = ar_model.AutoReg(serie, 1, trend='n')
        result = tsmodel.fit()
        ar_list.append(result.params[0])
    return np.array(ar_list)


def ARMA_11_phi_theta_parameters_estimation(series):
    """This function takes an array of ARMA(1,1) time series as an argument
        and returns  a list of the ar and ma estimated parameter for each series"""
    ar_list = []
    ma_list = []
    for serie in series:
        tsmodel = ARIMA(serie, order=(1, 0, 1), trend="n")
        result = tsmodel.fit()
        ar_list.append(result.params[0])
        ma_list.append(result.params[1])
    return np.array(ar_list), np.array(ma_list)


def ts_kurtosis_estimation(series_array, fisher=False, bias=False):
    """
    Calculate the kurtosis for each row in a 2D NumPy array.

    Parameters:
        series_array (numpy.ndarray): 2D array where each row is a vector.
        fisher (bool): if true, it calculates excess kurtosis
        bias (bool): if true, it calculates the biased estimator. Otherwise it applies correction.

    Returns:
        numpy array: A numpy array containing the kurtosis for each row.
    """
    return st.kurtosis(series_array, bias=bias, fisher=fisher, axis=1).flatten()
