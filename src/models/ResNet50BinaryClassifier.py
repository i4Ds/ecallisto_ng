import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from src.models.DefaultModel import DefaultBinaryModel


class ResNet50BinaryClassifier(DefaultBinaryModel):
    """
    Binary classifier based on ResNet50 architecture.
    """

    def __init__(self, lr=1e-3, weight_decay=1e-4):
        super().__init__(lr=lr, weight_decay=weight_decay)

        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(nn.Linear(num_features, 1), nn.Sigmoid())

        for param in self.resnet50.parameters():
            param.requires_grad = False

        layers_to_train = ["layer3", "layer4", "avgpool", "fc"]
        for name, child in self.resnet50.named_children():
            if name in layers_to_train:
                for param in child.parameters():
                    param.requires_grad = True

    def forward(self, x):
        return self.resnet50(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
