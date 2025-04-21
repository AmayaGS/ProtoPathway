import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    """Simple MLP baseline model for gene expression classification."""

    def __init__(self, input_size, hidden_size=512, num_classes=2, dropout_rate=0.1):
        """
        Args:
            input_size: Number of input features (genes)
            hidden_size: Size of hidden layers
            num_classes: Number of output classes
            dropout_rate: Dropout rate
        """
        super(MLPBaseline, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        # self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        # First layer
        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Second layer
        x = self.fc2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Output layer
        x = self.fc3(x)

        return x







class MLP_GE(torch.nn.Module):

    """"""

    def __init__(self, dim_in, num_classes):

        super().__init__()

        self.lin1 = torch.nn.Linear(dim_in, 512)
        self.lin2 = torch.nn.Linear(512, 206)
        self.lin3 = torch.nn.Linear(206, num_classes)

    def forward(self, x):

        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x_logits = self.lin3(x)
        x_out = F.softmax(x_logits, dim=1)

        return x_logits, x_out
