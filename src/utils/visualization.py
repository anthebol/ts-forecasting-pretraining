import matplotlib.pyplot as plt
import numpy as np


def plot_time_series_samples(data, num_samples=5, title="Time Series Samples"):
    """
    Plot multiple time series samples

    Args:
        data (np.ndarray): Time series data of shape (num_samples, seq_length)
        num_samples (int): Number of samples to plot
        title (str): Title for the plot
    """
    plt.figure(figsize=(15, 8))
    for i in range(min(num_samples, len(data))):
        plt.plot(data[i], label=f"Sample {i+1}", alpha=0.7)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_data_distribution(data, title="Data Distribution"):
    """
    Plot distribution of time series values

    Args:
        data (np.ndarray): Time series data
        title (str): Title for the plot
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(data.flatten(), bins=50, density=True)
    plt.title(f"{title}\nValue Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")

    plt.subplot(1, 2, 2)
    plt.boxplot(data.flatten())
    plt.title("Box Plot")

    plt.tight_layout()
    plt.show()


def plot_statistics(data, title="Time Series Statistics"):
    """
    Plot basic statistics of the time series

    Args:
        data (np.ndarray): Time series data of shape (num_samples, seq_length)
        title (str): Title for the plot
    """
    plt.figure(figsize=(15, 5))

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    plt.plot(mean, label="Mean", color="blue")
    plt.fill_between(
        range(len(mean)),
        mean - std,
        mean + std,
        alpha=0.2,
        color="blue",
        label="±1 STD",
    )

    plt.title(f"{title}\nMean and Standard Deviation")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
