from time_series_ddpm_model import TimeSeriesDDPM
from time_series_generator import TimeSeriesGenerator
import matplotlib.pyplot as plt
import torch


def train_and_plot():
    # Initialize the model
    model = TimeSeriesDDPM(input_dim=1, use_conv=True, conv_channels=16, num_lstm_layers=2, hidden_dim=32, T=1000)

    # Generate synthetic training data
    data = TimeSeriesGenerator.generate_ar_series(100, 100, 1)

    # Train the model
    model.train_model(data, num_epochs=100, plot=False)

    # Sample generated series
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    generated_series = model.sample(seq_length=data.shape[1], device=device, num_samples=2)
    print(generated_series)

    # Plotting the original and generated series
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot training data (first time series in the batch)
    ax[0].plot(data[0].cpu().numpy(), label="Training Series")
    ax[0].set_title("Training Time Series")
    ax[0].set_xlabel("Time Steps")
    ax[0].set_ylabel("Value")

    # Plot generated series
    ax[1].plot(generated_series[0], label="Generated Series", color='r')
    ax[1].set_title("Generated Time Series")
    ax[1].set_xlabel("Time Steps")
    ax[1].set_ylabel("Value")

    plt.tight_layout()
    plt.show()


def main():
    train_and_plot()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
