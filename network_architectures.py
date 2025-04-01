import torch
import torch.nn as nn

# 1. LSTM-based network
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        """
        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dim (int): Number of hidden units in the LSTM.
            num_layers (int): Number of LSTM layers.
        """
        super(LSTMNet, self).__init__()
        # The LSTM processes the input features concatenated with the time embedding.
        self.lstm = nn.LSTM(input_dim + 1, hidden_dim, num_layers=num_layers, batch_first=True)
        # Fully-connected layer projects LSTM output back to input_dim.
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        # Expand time embedding to match the sequence length and concatenate.
        t_expanded = t[:, None, None].expand(x.size(0), x.size(1), 1)
        x = torch.cat((x, t_expanded), dim=-1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


# 2. GRU-based network
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        """
        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dim (int): Number of hidden units in the GRU.
            num_layers (int): Number of GRU layers.
        """
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_dim + 1, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        t_expanded = t[:, None, None].expand(x.size(0), x.size(1), 1)
        x = torch.cat((x, t_expanded), dim=-1)
        x, _ = self.gru(x)
        x = self.fc(x)
        return x


# 3. Convolutional Neural Network (CNN)
class CNNNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=3, padding=1):
        """
        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dims (list of int): List defining the number of channels for each hidden convolutional layer.
            kernel_size (int): Kernel size for the convolution layers.
            padding (int): Padding for the convolution layers.
        """
        super(CNNNet, self).__init__()
        # When concatenating the time embedding, the number of channels increases by 1.
        in_channels = input_dim + 1
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Conv1d(in_channels, h_dim, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU())
            in_channels = h_dim
        # Build the CNN as a sequential model.
        self.cnn = nn.Sequential(*layers)
        # Fully-connected layer to map CNN output features back to input_dim.
        self.fc = nn.Linear(in_channels, input_dim)

    def forward(self, x, t):
        # x shape: (batch, seq, input_dim)
        # Expand time embedding and concatenate along the feature dimension.
        t_expanded = t[:, None, None].expand(x.size(0), x.size(1), 1)
        x = torch.cat((x, t_expanded), dim=-1)  # Shape: (batch, seq, input_dim+1)
        # Transpose to (batch, channels, seq) for CNN processing.
        x = x.transpose(1, 2)
        x = self.cnn(x)
        # Transpose back to (batch, seq, channels).
        x = x.transpose(1, 2)
        # Apply the fully-connected layer to each time step.
        x = self.fc(x)
        return x


# 4. Fully Connected Neural Network (FCNN)
class FCNNNet(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        """
        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dims (list of int): List defining the number of neurons for each hidden layer.
        """
        super(FCNNNet, self).__init__()
        layers = []
        in_features = input_dim + 1  # Add one for the time embedding.
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(nn.ReLU())
            in_features = h_dim
        # Final fully-connected layer to project back to input_dim.
        layers.append(nn.Linear(in_features, input_dim))
        self.fc_net = nn.Sequential(*layers)

    def forward(self, x, t):
        # x shape: (batch, seq, input_dim)
        t_expanded = t[:, None, None].expand(x.size(0), x.size(1), 1)
        x = torch.cat((x, t_expanded), dim=-1)
        batch, seq, _ = x.size()
        # Process each time step independently.
        x = x.view(batch * seq, -1)
        x = self.fc_net(x)
        x = x.view(batch, seq, -1)
        return x


# 5. Hybrid CNN-LSTM network: CNN preprocessing followed by LSTM layers
class HybridCNNLSTMNet(nn.Module):
    def __init__(self, input_dim, conv_channels, lstm_hidden_dim, num_lstm_layers=1, kernel_size=3, padding=1):
        """
        Args:
            input_dim (int): Dimensionality of the input features.
            conv_channels (int): Number of output channels for the CNN layer.
            lstm_hidden_dim (int): Number of hidden units in the LSTM.
            num_lstm_layers (int): Number of LSTM layers.
            kernel_size (int): Kernel size for the CNN layer.
            padding (int): Padding for the CNN layer.
        """
        super(HybridCNNLSTMNet, self).__init__()
        # CNN layer processes the raw input (without time).
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=conv_channels,
                              kernel_size=kernel_size, padding=padding)
        # LSTM processes the CNN output concatenated with the time embedding.
        self.lstm = nn.LSTM(conv_channels + 1, lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True)
        # Fully-connected layer projects LSTM output back to input_dim.
        self.fc = nn.Linear(lstm_hidden_dim, input_dim)

    def forward(self, x, t):
        # Apply CNN: transpose to (batch, input_dim, seq) and process.
        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # Now shape: (batch, seq, conv_channels)
        # Concatenate time embedding.
        t_expanded = t[:, None, None].expand(x_conv.size(0), x_conv.size(1), 1)
        x_combined = torch.cat((x_conv, t_expanded), dim=-1)
        # Process through LSTM.
        x_lstm, _ = self.lstm(x_combined)
        # Project to input_dim.
        x_out = self.fc(x_lstm)
        return x_out
