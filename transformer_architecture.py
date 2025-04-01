import math
import torch
import torch.nn as nn

# TODO: this has been fully coded by chatGPT. Since I don't fully understand transformer architectures,
#  this code should be treated carefully. The results are terrible and the training loss is greater
#  than 1 in the first epochs, which likely means there is something that we are doing wrong here

# Positional Encoding module (optional)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Args:
            d_model (int): Embedding dimension.
            dropout (float): Dropout probability.
            max_len (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the div_term for even indices.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sin to even indices in the positional encoding.
        pe[:, 0::2] = torch.sin(position * div_term)
        # For cosine, ensure that we only take as many values from div_term as needed.
        odd_dim = pe[:, 1::2].size(1)
        pe[:, 1::2] = torch.cos(position * div_term[:odd_dim])

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor: Output tensor with positional encoding applied.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# Transformer-based network
class TransformerNet(nn.Module):
    def __init__(
            self,
            input_dim,
            embed_dim,
            num_layers=1,
            nhead=1,
            dim_feedforward=2048,
            dropout=0.1,
            use_positional_encoding=True,
    ):
        """
        Args:
            input_dim (int): Dimensionality of the input features.
            embed_dim (int): Embedding dimension for the transformer.
            num_layers (int): Number of transformer encoder layers.
            nhead (int): Number of attention heads.
            dim_feedforward (int): Dimensionality of the feedforward network in the transformer.
            dropout (float): Dropout probability.
            use_positional_encoding (bool): Whether to use positional encoding.
        """
        super(TransformerNet, self).__init__()
        # The input features are augmented with a time embedding, so input becomes (input_dim + 1)
        self.input_proj = nn.Linear(input_dim + 1, embed_dim)
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        else:
            self.positional_encoding = None

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully-connected layer projects the transformer output back to input_dim.
        self.fc = nn.Linear(embed_dim, input_dim)

    def forward(self, x, t):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, seq, input_dim).
            t (Tensor): Time tensor of shape (batch,).
        Returns:
            Tensor: Output tensor of shape (batch, seq, input_dim).
        """
        # Expand t from (batch,) to (batch, seq, 1)
        t_expanded = t[:, None, None].expand(x.size(0), x.size(1), 1)
        # Concatenate the time embedding to the input features.
        x = torch.cat((x, t_expanded), dim=-1)  # Shape: (batch, seq, input_dim+1)
        # Project to the transformer embedding dimension.
        x = self.input_proj(x)  # Shape: (batch, seq, embed_dim)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        # Process with transformer encoder.
        x = self.transformer_encoder(x)  # Shape: (batch, seq, embed_dim)
        # Project back to the original input_dim.
        x = self.fc(x)  # Shape: (batch, seq, input_dim)
        return x