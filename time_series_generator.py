import torch
import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample


class TimeSeriesGenerator:
    @staticmethod
    def generate_arma_series(num_series, seq_length, ar, ma, sigma=0.1, burnin=100):
        """
        Generate ARMA(length(ar), length(ma)) series
        ar: array_like
            The coefficient for autoregressive lag polynomial, including zero lag.

        ma: array_like
            The coefficient for moving-average lag polynomial, including zero lag.
        """

        ar = np.r_[1, -ar]  # add zero-lag and negate
        ma = np.r_[1, ma]  # add zero-lag

        # Initialize array where we store all the series
        series_nparray = np.empty((num_series, seq_length))

        # Fill each row with an ARMA sample
        for i in range(num_series):
            series_nparray[i, :] = arma_generate_sample(ar, ma, seq_length, scale=sigma, burnin=burnin)
        return torch.tensor(series_nparray, dtype=torch.float32).unsqueeze(-1)

    @staticmethod
    def generate_ar_series(num_series, seq_length, ar, sigma=0.1, burnin=100):
        return TimeSeriesGenerator.generate_arma_series(num_series, seq_length, ar, ma=0, sigma=sigma, burnin=burnin)

    #TODO
    @staticmethod
    def generate_arima_series(num_series=100, seq_length=100, phi=0.8, theta=0.5, sigma=0.1, d=1):
        raise NotImplementedError("Function not yet developed")

    @staticmethod
    def generate_garch_11_series(num_series=100, seq_length=100, omega=0.1, alpha=0.15, beta=0.8, burnin=100):
        """
        Generate GARCH(1,1) series using vectorized numpy operations:
            x[t] = sigma_t * e[t]    with e[t] ~ N(0,1)
            sigma_t^2 = omega + alpha * x[t-1]^2 + beta * sigma_{t-1}^2

        A burn-in period can be specified to let the process reach stationarity.

        The initial variance is set to the unconditional variance omega/(1 - alpha - beta)
        when the process is stationary (alpha+beta < 1), otherwise it is set to 1.0.

        Parameters:
            num_series (int): Number of independent series to generate.
            seq_length (int): Length of the recorded series (after burn-in).
            omega (float): Constant term in the variance equation.
            alpha (float): Coefficient for lagged squared return.
            beta (float): Coefficient for lagged variance.
            burnin (int): Number of initial steps to discard.

        Returns:
            torch.Tensor: Tensor of shape (num_series, seq_length, 1) containing the generated series.
        """
        total_length = seq_length + burnin
        # Preallocate the array for all series
        x = np.empty((num_series, total_length))

        # Set the initial variance for each series
        if (alpha + beta) < 1:
            init_sigma2 = omega / (1 - alpha - beta)
        else:
            init_sigma2 = 1.0
        sigma2 = np.full(num_series, init_sigma2)

        # Simulate the series for each time step, vectorizing over the number of series.
        for t in range(total_length):
            sigma_t = np.sqrt(sigma2)
            e_t = np.random.randn(num_series)
            x[:, t] = sigma_t * e_t
            # Update variance for the next time step
            sigma2 = omega + alpha * (x[:, t] ** 2) + beta * sigma2

        # Discard the burn-in period and convert to a PyTorch tensor.
        x = x[:, burnin:]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1)

