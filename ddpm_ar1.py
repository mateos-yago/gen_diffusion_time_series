#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 22:02:33 2025

@author: xing
"""
    
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

class TimeSeriesDDPM(nn.Module):
    """
    A Diffusion Denoising Probabilistic Model (DDPM) for time series data that uses an LSTM-based architecture.

    Attributes:
        lstm (nn.LSTM): LSTM layer for sequence processing with an extra input dimension for time embedding.
        fc (nn.Linear): Fully-connected layer mapping LSTM hidden states to the input dimension.
        T (int): Total number of diffusion timesteps.
        betas (torch.Tensor): Linear schedule of noise variance parameters.
        alphas (torch.Tensor): Complement of betas (i.e., 1 - beta).
        alphas_cumprod (torch.Tensor): Cumulative product of alphas over timesteps.
    """

    def __init__(self, input_dim, hidden_dim, T=1000):
        """
        Initialize the TimeSeriesDDPM model.

        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dim (int): Number of hidden units in the LSTM layer.
            T (int, optional): Total number of diffusion timesteps. Defaults to 1000.

        This constructor sets up:
            - An LSTM layer that accepts input_dim + 1 features (the extra one is for the time embedding).
            - A fully-connected layer to project the LSTM output back to input_dim.
            - The noise schedule parameters (betas, alphas, alphas_cumprod) used for the diffusion process.
        """
        super(TimeSeriesDDPM, self).__init__()
        self.lstm = nn.LSTM(input_dim + 1, hidden_dim, batch_first=True)  # +1 for time embedding
        self.fc = nn.Linear(hidden_dim, input_dim)

        # Noise schedule parameters
        self.T = T
        self.betas = torch.linspace(1e-4, 0.02, T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward(self, x, t):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): Input time series tensor of shape (batch_size, sequence_length, input_dim).
            t (torch.Tensor): Tensor of time step indices with shape (batch_size,).

        Returns:
            torch.Tensor: Predicted output tensor of shape (batch_size, sequence_length, input_dim).

        The method expands the time tensor to match the sequence length, concatenates it to the input,
        processes the combined tensor with the LSTM, and then applies a fully-connected layer.
        """
        t = t[:, None, None].expand(-1, x.shape[1], 1)  # Expand t to match sequence shape
        x = torch.cat((x, t), dim=-1)  # Concatenate t to input
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

    def q_sample(self, x0, t):
        """
        Sample from the forward diffusion process at timestep t.

        Args:
            x0 (torch.Tensor): The original (clean) time series tensor.
            t (torch.Tensor): Tensor of time step indices for each sample in the batch.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The noisy version of x0 at timestep t.
                - torch.Tensor: The noise tensor that was added to x0.

        The method generates Gaussian noise and mixes it with x0 using the precomputed noise schedule
        (alphas_cumprod) for the given timesteps.
        """
        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = torch.sqrt(self.alphas_cumprod[t])[:, None, None]
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None]
        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise, noise

    def p_sample(self, xt, t):
        """
        Sample from the reverse diffusion process (denoising step) at timestep t.

        Args:
            xt (torch.Tensor): The noisy time series tensor at timestep t.
            t (int or torch.Tensor): The current timestep index. For a single sample, an integer is expected.

        Returns:
            torch.Tensor: The denoised tensor corresponding to the previous timestep.

        The method uses the model's forward pass to predict the noise component and then applies the reverse
        diffusion equation to generate a less noisy sample. A new noise term is added if t > 0.
        """
        with torch.no_grad():
            noise_pred = self.forward(xt, t)
            beta_t = self.betas[t][:, None, None]
            alpha_t = self.alphas[t][:, None, None]
            sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

            if t > 0:
                noise = torch.randn_like(xt)
            else:
                noise = 0

            return sqrt_recip_alpha_t * (xt - beta_t / sqrt_one_minus_alpha_t * noise_pred) + noise

    def train_model(self, data, num_epochs=5, batch_size=10, learning_rate=1e-3):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            for x0_batch in dataloader:
                x0 = x0_batch[0].to(device)
                # sample random time step
                t = torch.randint(0, self.T, (x0.shape[0],), device=device)
                # add noise to the batch according to given time step.
                xt, noise = self.q_sample(x0, t)

                optimizer.zero_grad()
                noise_pred = self.forward(xt, t)
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    def train_model(self, data, num_epochs=5, batch_size=10, learning_rate=1e-3, plot=False):
        """
        Train the TimeSeriesDDPM model using provided time series data.

        Args:
            data (torch.Tensor): Training data tensor of shape (num_series, sequence_length, input_dim).
            num_epochs (int, optional): Number of training epochs. Defaults to 5.
            batch_size (int, optional): Batch size for training. Defaults to 10.
            learning_rate (float, optional): Learning rate for the Adam optimizer. Defaults to 1e-3.
            plot (bool, optional): If True, plot predicted noise vs. actual noise for the first sample in the last batch. Defaults to False.

        Returns:
            None

        The function:
            - Sets up the training device and optimizer.
            - Wraps the data in a DataLoader for batching.
            - For each epoch, samples random timesteps, generates noisy data via q_sample,
              computes the predicted noise using the forward pass, and minimizes the MSE loss.
            - Prints the loss after each epoch.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            epoch_loss = 0
            last_noise = None
            last_noise_pred = None
            for x0_batch in dataloader:
                x0 = x0_batch[0].to(device)
                t = torch.randint(0, self.T, (x0.shape[0],), device=device)
                xt, noise = self.q_sample(x0, t)

                optimizer.zero_grad()
                noise_pred = self(xt, t)
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * x0.size(0)
                # Save the last batch's noise for plotting later (take the first sample in the batch)
                last_noise = noise[0]
                last_noise_pred = noise_pred[0]
            epoch_loss /= len(dataset)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

            if plot:

                # Plot predicted noise vs. actual noise for the first sample of the last batch
                noise_np = last_noise.detach().cpu().numpy().squeeze()      # shape: [sequence_length]
                noise_pred_np = last_noise_pred.detach().cpu().numpy().squeeze()  # shape: [sequence_length]

                plt.figure()
                plt.plot(noise_np, label="Actual Noise")
                plt.plot(noise_pred_np, label="Predicted Noise", linestyle="--")
                plt.title(f"Epoch {epoch+1}: Predicted vs. Actual Noise")
                plt.xlabel("Time Steps")
                plt.ylabel("Noise Value")
                plt.legend()
                plt.show()

    def sample(self, seq_length, device):
        """
        Generate a new time series by reversing the diffusion process.

        Args:
            seq_length (int): Length of the time series sequence to generate.
            device (torch.device): Device on which to perform computation (CPU or GPU).

        Returns:
            numpy.ndarray: The generated time series as a NumPy array with shape (1, seq_length, input_dim).

        The method begins with a tensor of pure noise and iteratively applies the reverse diffusion
        (p_sample) for each timestep from T-1 to 0.
        """
        self.eval()
        x = torch.randn((1, seq_length, 1), device=device)  # Start with noise
        for t in reversed(range(self.T)):
            x = self.p_sample(x, torch.full((1,), t, device=device, dtype=torch.long))
        return x.cpu().numpy()

#######################################################################
# Generate AR(1) synthetic time series
def generate_ar1_series(num_series=100, seq_length=100, phi=0.8, sigma=0.1):
    series = []
    for _ in range(num_series):
        x = [np.random.randn()]
        for t in range(1, seq_length):
            x.append(phi * x[-1] + sigma * np.random.randn())
        series.append(x)
    return torch.tensor(series, dtype=torch.float32).unsqueeze(-1)  # Add input_dim=1




#######################################################################
# Function to train the model and visualize results
def train_and_plot():
    # Initialize the model
    model = TimeSeriesDDPM(input_dim=1, hidden_dim=32, T=1000)
    
    # Generate synthetic training data
    data = generate_ar1_series()
    
    # Train the model
    model.train_model(data, num_epochs=5)
    
    # Sample generated series
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    generated_series = model.sample(seq_length=data.shape[1], device=device)
    
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

# Call the function to train, generate, and plot
train_and_plot()




