import torch
import torch.nn as nn
import torchvision.models as models
from src.models.DefaultModel import DefaultBinaryModel

class SqueezeNet(DefaultBinaryModel):
    def __init__(self, lr=1e-3, weight_decay=1e-4, dropout_p=0.0):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.dropout_p = dropout_p
        
        # Load the SqueezeNet model
        self.squeezenet = models.squeezenet1_1(weights="DEFAULT")
        
        # Replace the classifier to match the number of output classes
        self.squeezenet.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_p),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.squeezenet(x)
        x = torch.sigmoid(x)
        return torch.flatten(x, 1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
