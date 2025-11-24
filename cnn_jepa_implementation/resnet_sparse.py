import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

class SparseResNet(resnet.ResNet):
    def __init__(self, block, layers, **kwargs):
        super().__init__(block, layers, **kwargs)
        # Remove fc layer as we need spatial features
        self.fc = nn.Identity()
        self.avgpool = nn.Identity()

    def _apply_mask(self, x, mask):
        if mask is None:
            return x
        
        B, C, H, W = x.shape
        # Mask is (B, 1, H_mask, W_mask)
        # Resize mask to (B, 1, H, W) using nearest neighbor
        m = F.interpolate(mask.float(), size=(H, W), mode='nearest')
        return x * m

    def _forward_impl(self, x, mask=None):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self._apply_mask(x, mask)
        
        x = self.maxpool(x)
        x = self._apply_mask(x, mask)

        # Stages
        x = self.layer1(x)
        x = self._apply_mask(x, mask)
        
        x = self.layer2(x)
        x = self._apply_mask(x, mask)
        
        x = self.layer3(x)
        x = self._apply_mask(x, mask)
        
        x = self.layer4(x)
        x = self._apply_mask(x, mask)

        # Return spatial features (B, 2048, H/32, W/32)
        return x

    def forward(self, x, mask=None):
        return self._forward_impl(x, mask)

def sparse_resnet50(**kwargs):
    model = SparseResNet(resnet.Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
