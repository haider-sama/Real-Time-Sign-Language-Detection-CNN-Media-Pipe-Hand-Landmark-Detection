from torch.nn import (
    Linear, ReLU, Sequential, Conv1d, MaxPool1d, Module, BatchNorm1d, Dropout
)

class CNNModel(Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.cnnLayers = Sequential(
            # First convolutional block
            Conv1d(in_channels=63, out_channels=32, kernel_size=3, stride=1, padding=2),
            BatchNorm1d(32),
            ReLU(),

            # Second convolutional block
            Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            BatchNorm1d(64),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),

            # Third convolutional block
            Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
            BatchNorm1d(128),
            ReLU(),
            Dropout(p=0.3),

            # Fourth convolutional block
            Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=2),
            BatchNorm1d(256),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),

            # Fifth convolutional block
            Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2),
            BatchNorm1d(512),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),

            # Sixth convolutional block
            Conv1d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2),
            BatchNorm1d(512),
            ReLU(),
            Dropout(p=0.3),
        )

        # Linear layers
        self.linearLayers = Sequential(
            Linear(in_features=512, out_features=26),
            BatchNorm1d(26),
            ReLU(),
        )

    def forward(self, x):
        """Forward pass"""
        # Pass input through convolutional layers
        x = self.cnnLayers(x)

        # Flatten the tensor for the linear layers
        x = x.view(x.size(0), -1)

        # Pass flattened tensor through linear layers
        x = self.linearLayers(x)

        return x
