import numpy as np


class Metrics:
    """Time series forecasting evaluation metrics"""

    @staticmethod
    def MSE(pred, true):
        return np.mean((pred - true) ** 2)

    @staticmethod
    def MAE(pred, true):
        return np.mean(np.abs(pred - true))

    @staticmethod
    def SMAPE(pred, true):
        """Symmetric Mean Absolute Percentage Error"""
        return 100 * np.mean(2.0 * np.abs(pred - true) / (np.abs(pred) + np.abs(true)))

    @staticmethod
    def MASE(pred, true, history):
        """Mean Absolute Scaled Error"""
        n = len(true)
        d = np.abs(np.diff(history)).mean()  # scale factor
        errors = np.abs(pred - true)
        return errors.mean() / d

    @staticmethod
    def evaluate_all(pred, true, history=None):
        metrics = {
            "MSE": Metrics.MSE(pred, true),
            "MAE": Metrics.MAE(pred, true),
            "SMAPE": Metrics.SMAPE(pred, true),
        }
        if history is not None:
            metrics["MASE"] = Metrics.MASE(pred, true, history)
        return metrics
