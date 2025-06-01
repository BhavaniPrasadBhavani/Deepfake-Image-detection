import torch.nn as nn
import torchvision.models as models

class DeepfakeResNet(nn.Module):
    def __init__(self):
        super(DeepfakeResNet, self).__init__()
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True)
        # Modify the final layer for binary classification
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x) 