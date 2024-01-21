import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from src.models.DefaultModel import DefaultBinaryModel
from torchvision.models.efficientnet import efficientnet_v2_s


class EfficientNetV2SBinaryClassifier(DefaultBinaryModel):
    """
    Binary classifier based on EfficientNetV2S architecture.
    """

    def __init__(self, lr=1e-3, weight_decay=1e-4):
        super().__init__(lr=lr, weight_decay=weight_decay)

        # EfficientNet V2 S Modell laden
        self.efficientnet_v2_s = efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        num_features = self.efficientnet_v2_s.classifier[1].in_features
        self.efficientnet_v2_s.classifier = nn.Sequential(nn.Linear(num_features, 1), nn.Sigmoid())

        # Layer einfrieren
        for param in self.efficientnet_v2_s.parameters():
            param.requires_grad = False

        # Submodule zum Trainieren bestimmen (Beispiel: die letzten drei Submodule von 'features')
        submodules_to_train = ["5", "6", "7"]
        for name, module in self.efficientnet_v2_s.named_children():
            if name == "features":
                for sub_name, sub_module in module.named_children():
                    if sub_name in submodules_to_train:
                        for param in sub_module.parameters():
                            param.requires_grad = True
            elif name in ["avgpool", "classifier"]:
                for param in module.parameters():
                    param.requires_grad = True

    def forward(self, x):
        return self.efficientnet_v2_s(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
