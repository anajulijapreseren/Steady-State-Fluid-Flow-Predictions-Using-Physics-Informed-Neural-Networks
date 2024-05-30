import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Network(nn.Module):
    def __init__(self, num_inputs=2, num_layers=8, neurons_per_layer=40, num_outputs=2,
                 activation='tanh', dropout_rate=0.0, use_skip_connections=False, init_type='xavier'):
        """
        Initialize the neural network with a configurable number of layers and neurons per layer,
        and optionally add dropout or skip connections.

        Args:
            num_inputs (int): Number of input features, e.g., 2 for x, y coordinates.
            num_layers (int): Number of hidden layers in the network.
            neurons_per_layer (int): Number of neurons in each hidden layer.
            num_outputs (int): Number of output features, e.g., 2 for psi, p outputs.
            activation (str): Activation function to use ('relu', 'tanh', 'sigmoid').
            dropout_rate (float): Dropout rate between 0 and 1, 0 means no dropout.
            use_skip_connections (bool): If True, adds skip connections between some layers.
        """
        super(Network, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.activation_func = self.get_activation_func(activation)
        self.init_type = init_type 

        # Create layers
        current_dim = num_inputs
        for _ in range(num_layers):
            layer = nn.Linear(current_dim, neurons_per_layer)
            self.layers.append(layer)
            self.init_weights(layer)  # Custom weight initialization
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
            current_dim = neurons_per_layer
        
        # Output layer
        output_layer = nn.Linear(neurons_per_layer, num_outputs)
        self.layers.append(output_layer)
        self.init_weights(output_layer) 
        
        # Skip connections: store indices where skip connections will be added
        self.skip_layers={}
        if use_skip_connections:
            self.skip_layers = {i: i + 2 for i in range(0, num_layers, 2) if i + 2 < num_layers}

    def init_weights(self, layer):
        """Initialize weights and biases for each layer."""
        if self.init_type == 'kaiming':
            init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        elif self.init_type == 'xavier':
            init.xavier_uniform_(layer.weight)
        else:
            raise ValueError("Unsupported initialization type")

        if layer.bias is not None:
            init.constant_(layer.bias, 0)


    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor containing the input features.
        
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        identity = x
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if i in self.skip_layers:
                    x += identity  # Add skip connection
                    identity = x  # Update identity
                x = self.activation_func(x)
            else:
                x = layer(x)  # Apply dropout
        return x

    def get_activation_func(self, activation):
        """ Return the activation function based on the input string """
        if activation == 'relu':
            return F.relu
        elif activation == 'tanh':
            return torch.tanh
        elif activation == 'sigmoid':
            return torch.sigmoid
        else:
            raise ValueError(f"Unsupported activation function '{activation}'")

