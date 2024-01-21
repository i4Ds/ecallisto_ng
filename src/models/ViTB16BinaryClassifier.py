import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import vit_b_16
from src.models.DefaultModel import DefaultBinaryModel


class ViTB16BinaryClassifier(DefaultBinaryModel):
    """
    Binary classifier based on Vision Transformer Base 16 architecture.
    """

    def __init__(self, lr=1e-4, weight_decay=1e-4):
        super().__init__(lr=lr, weight_decay=weight_decay)

        # ViT B-16 Modell laden
        self.vit_b_16 = vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

        # Parameter freezen
        for param in self.vit_b_16.parameters():
            param.requires_grad = False

        # Die letzten Encoder-Layer unfreezen
        for layer in [self.vit_b_16.encoder.layers.encoder_layer_10, self.vit_b_16.encoder.layers.encoder_layer_11]:
            for param in layer.parameters():
                param.requires_grad = True

        # Output zu binär umwandeln
        self.vit_b_16.heads = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())

    def forward(self, x):
        return self.vit_b_16(x)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
