import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from src.models.DefaultModel import DefaultBinaryModel


class ResNet18BinaryClassifier(DefaultBinaryModel):
    """
    Binary classifier based on ResNet18 architecture.
    """

    def __init__(self, lr=1e-3, weight_decay=1e-4):
        super().__init__(lr=lr, weight_decay=weight_decay)

        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(nn.Linear(num_features, 1), nn.Sigmoid())

        for param in self.resnet18.parameters():
            param.requires_grad = False

        layers_to_train = ["layer3", "layer4", "avgpool", "fc"]
        for name, child in self.resnet18.named_children():
            if name in layers_to_train:
                for param in child.parameters():
                    param.requires_grad = True

    def forward(self, x):
        return self.resnet18(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
