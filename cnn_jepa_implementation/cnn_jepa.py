import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from resnet_sparse import sparse_resnet50

class PredictorBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Depthwise Separable Conv
        # 1. Depthwise: groups=channels
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        # 2. Pointwise: 1x1
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CNNJEPA(nn.Module):
    def __init__(self, feature_dim=2048):
        super().__init__()
        
        # Context Encoder
        self.context_encoder = sparse_resnet50()
        
        # Target Encoder
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
            
        # Predictor
        self.predictor = nn.Sequential(
            PredictorBlock(feature_dim),
            PredictorBlock(feature_dim),
            PredictorBlock(feature_dim)
        )
        
        # Mask Token
        self.mask_token = nn.Parameter(torch.randn(1, feature_dim, 1, 1))
        
    def update_target_encoder(self, momentum):
        with torch.no_grad():
            for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(momentum).add_((1.0 - momentum) * param_q.data)

    def forward(self, images, masks):
        """
        Args:
            images: (B, 3, H, W)
            masks: (B, 1, H_feat, W_feat) - 1 for context, 0 for target
        """
        # 1. Context Encoding
        # masks need to be resized inside the encoder, but here we pass the feature-map aligned mask
        context_features = self.context_encoder(images, masks)
        
        # 2. Target Encoding
        with torch.no_grad():
            target_features = self.target_encoder(images, mask=None)
            
        # 3. Token Filling
        # masks is (B, 1, H_feat, W_feat). 0 indicates masked region (target).
        # We want to replace masked regions with mask_token.
        # Ensure mask is broadcastable
        mask_bool = masks.bool() # (B, 1, H, W)
        
        # Expand mask token to batch size
        B, C, H, W = context_features.shape
        # mask_token is (1, C, 1, 1)
        
        # Fill: where mask is 0 (False), use mask_token. Where mask is 1 (True), keep context.
        # Note: context_features should already be 0 at masked locations due to sparse encoder,
        # but adding (1-mask)*token is safer/cleaner.
        # Or just use torch.where
        
        filled_features = torch.where(mask_bool, context_features, self.mask_token)
        
        # 4. Prediction
        predicted_features = self.predictor(filled_features)
        
        # 5. Loss Calculation
        # We only compute loss on masked regions (where mask == 0)
        # Extract features at masked locations
        # mask_bool is True for Context, False for Target.
        # We want Target regions.
        target_mask = ~mask_bool
        
        # Flatten spatial dimensions for easier indexing if needed, or just mask
        # predicted_features: (B, C, H, W)
        # target_features: (B, C, H, W)
        
        # We can just mask out the non-target regions and compute sum/mean, 
        # but standard practice is to extract the elements.
        
        pred_masked = predicted_features.permute(0, 2, 3, 1)[target_mask.squeeze(1)] # (N_masked, C)
        target_masked = target_features.permute(0, 2, 3, 1)[target_mask.squeeze(1)]   # (N_masked, C)
        
        loss = F.mse_loss(pred_masked, target_masked)
        
        return loss
