import torch.nn as nn
import torchvision.models as models

class DeepfakeResNet(nn.Module):
    def __init__(self):
        super(DeepfakeResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)  # Real or Fake

    def forward(self, x):
        return self.resnet(x) 