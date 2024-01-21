import torch
import torch.nn as nn
from src.models.DefaultModel import DefaultBinaryModel

class CustomCNN(DefaultBinaryModel):
    def __init__(self, lr=1e-3, weight_decay=1e-4, dropout_p=0.0):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.dropout_p = dropout_p
        
        # Increased convolutional layers with more filters
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=1, padding=1)

        # Smaller linear layers
        self.fc1 = nn.Linear(512 * 12 * 14, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x):
        x = x[:, 0].unsqueeze(1)
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        x = x.flatten(start_dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class CustomCNN2(DefaultBinaryModel):
    def __init__(self, lr=1e-3, weight_decay=1e-4, dropout_p=0.0):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.dropout_p = dropout_p
        
        # Reduced convolutional layers filters for complexity reduction
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization after Conv1
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization after Conv2
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128) # Batch Normalization after Conv3
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256) # Batch Normalization after Conv4

        # Adjusted linear layer sizes
        self.fc1 = nn.Linear(256 * 12 * 14, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x):
        x = x[:, 0].unsqueeze(1)
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool(nn.functional.relu(self.bn3(self.conv3(x))))
        x = self.pool(nn.functional.relu(self.bn4(self.conv4(x))))
        x = x.flatten(start_dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

