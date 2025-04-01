import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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

    def __init__(self, input_dim, hidden_dim, use_conv=False, conv_channels=None, num_lstm_layers=1, T=1000):
        """
        Initialize the TimeSeriesDDPM model.

        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dim (int): Number of hidden units in the LSTM layer.
            use_conv (bool): Whether to use nor not a convolutional 1D preprocessing layer
            conv_channels(int): Number of output channels for the optional convolutional layer
            num_lstm_layers(int): number of hidden lstm layers.
            T (int, optional): Total number of diffusion timesteps. Defaults to 1000.

        This constructor sets up:
            - An LSTM layer that accepts input_dim + 1 features (the extra one is for the time embedding).
            - A fully-connected layer to project the LSTM output back to input_dim.
            - The noise schedule parameters (betas, alphas, alphas_cumprod) used for the diffusion process.
        """
        super(TimeSeriesDDPM, self).__init__()
        self.use_conv = use_conv

        if self.use_conv:
            if conv_channels is None:
                raise ValueError("conv_channels must be specified when use_conv is True")
            # For conv1d: input shape (batch, channels, seq_length)
            self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv_channels,
                                   kernel_size=3, padding=1)
            lstm_input_dim = conv_channels + 1  # plus one for time embedding
        else:
            lstm_input_dim = input_dim + 1  # original features plus time embedding

        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=num_lstm_layers, batch_first=True)
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
        if self.use_conv:
            # Transpose to (batch, input_dim, seq_length) for conv1d
            x = x.transpose(1, 2)
            x = self.conv1(x)
            # Transpose back to (batch, seq_length, conv_channels)
            x = x.transpose(1, 2)

        # a partir de aqui es el codigo original
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
        with (torch.no_grad()):
            noise_pred = self.forward(xt, t)
            beta_t = self.betas[t][:, None, None]
            alpha_t = self.alphas[t][:, None, None]
            sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)
            # Use cumulative product for the scaling:
            alpha_bar_t = self.alphas_cumprod[t][:, None, None]
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

            noise = torch.where(t.view(-1, 1, 1) > 0,
                                torch.randn_like(xt),
                                torch.zeros_like(xt))

            return sqrt_recip_alpha_t * (xt - beta_t / sqrt_one_minus_alpha_bar_t * noise_pred) + \
                noise*torch.sqrt(beta_t)


    def sample(self, seq_length, device, num_samples=1):
        """
        Generate a new time series by reversing the diffusion process.

        Args:
            seq_length (int): Length of the time series sequence to generate.
            device (torch.device): Device on which to perform computation (CPU or GPU).
            num_samples (int): Number of series to generate (default is 1).

        Returns:
            numpy.ndarray: The generated time series as a NumPy array with shape (num_samples, seq_length, input_dim).

        The method begins with a tensor of pure noise and iteratively applies the reverse diffusion
        (p_sample) for each timestep from T-1 to 0.
        """
        self.eval()
        # Initialize with noise for num_samples samples
        x = torch.randn((num_samples, seq_length, 1), device=device)
        for t in reversed(range(self.T)):
            # Create a tensor of time indices for all samples
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_tensor)
        return x.cpu().numpy()
