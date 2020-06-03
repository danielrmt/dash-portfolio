import numpy as np
import pandas as pd


class MeanVariancePortfolio:
    def __init__(self, expected_returns, covariance_matrix, tickers):
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.tickers = tickers

    def portfolio_return(self, weights):
        return self.expected_returns.T @ weights

    def portfolio_covariance(self, weights):
        return weights.T @ self.covariance_matrix @ weights

    def frontier(self, min=0, max=.05, step=.0001):
        sig_inv = np.linalg.inv(self.covariance_matrix)
        ones = np.repeat(1, self.expected_returns.shape)
        A = ones.T @ sig_inv @ ones
        B = ones.T @ sig_inv @ self.expected_returns
        C = self.expected_returns.T @ sig_inv @ self.expected_returns
        D = A * C - B ** 2
        df = pd.DataFrame(np.arange(min, max, step), columns=['mu'])
        df['sigma'] = np.sqrt((A * df['mu']**2 - 2 * B * df['mu'] + C) / D)

        w_min = sig_inv @ ones / A
        w_tan = sig_inv @ self.expected_returns / B
        
        df['lambda'] = (C - df['mu'] * B) / D
        df['gamma'] = (df['mu'] * A - B) / D
        i = 0
        for col in self.tickers:
            w = df['lambda'] * A * w_min[i] + df['gamma'] * B * w_tan[i]
            df[col] = w
            i = i + 1

        return df
