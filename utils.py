from statsmodels.tsa import ar_model

def AR_1_phi_parameter_estimation(series):
    """This function takes an array of AR1 time series as an argument
    and returns  a list of the phi1 estimated parameter for each series"""
    params_list=[]
    for serie in series:
        tsmodel=ar_model.AutoReg(serie, 1, trend='n')
        result=tsmodel.fit()
        params_list.append(result.params[0])
    return params_list