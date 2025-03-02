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
    def __init__(self, input_dim, hidden_dim, T=1000):
        super(TimeSeriesDDPM, self).__init__()
        self.lstm = nn.LSTM(input_dim + 1, hidden_dim, batch_first=True)  # +1 for time embedding
        self.fc = nn.Linear(hidden_dim, input_dim)
        
        # Noise schedule parameters
        self.T = T
        self.betas = torch.linspace(1e-4, 0.02, T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def forward(self, x, t):
        t = t[:, None, None].expand(-1, x.shape[1], 1)  # Expand t to match sequence shape
        x = torch.cat((x, t), dim=-1)  # Concatenate t to input
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

    # Noise schedule functions
    def q_sample(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = torch.sqrt(self.alphas_cumprod[t])[:, None, None]
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None]
        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise, noise
    
    def p_sample(self, xt, t):
        with torch.no_grad():
            noise_pred = self(xt, t)
            beta_t = self.betas[t][:, None, None]
            alpha_t = self.alphas[t][:, None, None]
            sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            
            if t > 0:
                noise = torch.randn_like(xt)
            else:
                noise = 0
            
            return sqrt_recip_alpha_t * (xt - beta_t / sqrt_one_minus_alpha_t * noise_pred) + noise

    # Training function
    def train_model(self, data, num_epochs=5, batch_size=10, learning_rate=1e-3):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            for x0_batch in dataloader:
                x0 = x0_batch[0].to(device)
                t = torch.randint(0, self.T, (x0.shape[0],), device=device)
                xt, noise = self.q_sample(x0, t)
                
                optimizer.zero_grad()
                noise_pred = self(xt, t)
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Sampling function
    def sample(self, seq_length, device):
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




