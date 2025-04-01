from statsmodels.tsa import ar_model
from statsmodels.tsa.arima.model import ARIMA
import scipy.stats as st

def AR_1_phi_parameter_estimation(series):
    """This function takes an array of AR(1) time series as an argument
    and returns  a list of the phi1 estimated parameter for each series"""
    params_list=[]
    for serie in series:
        tsmodel=ar_model.AutoReg(serie, 1, trend='n')
        result=tsmodel.fit()
        params_list.append(result.params[0])
    return params_list

def ARMA_11_phi_theta_parameters_estimation(series):
    """This function takes an array of ARMA(1,1) time series as an argument
        and returns  a list of the phi and theta estimated parameter for each series"""
    phi_list=[]
    theta_list=[]
    for serie in series:
        tsmodel=ARIMA(serie, order=(1, 0, 1), trend="n")
        result=tsmodel.fit()
        phi_list.append(result.params[0])
        theta_list.append(result.params[1])
    return phi_list, theta_list

def ts_kurtosis_estimation(series):
    kurtosis_list=[]
    for serie in series:
        kurtosis_list.append(st.kurtosis(serie))
    return kurtosis_list
