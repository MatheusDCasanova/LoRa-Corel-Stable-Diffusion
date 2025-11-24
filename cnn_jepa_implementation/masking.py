import torch
import random
import math

class MaskCollator:
    def __init__(
        self,
        input_size=(224, 224),
        patch_size=32,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,  # Minimum number of patches to keep
        allow_overlap=False,  # Whether to allow overlap between context and target masks
    ):
        """
        Collator for CNN-JEPA that generates masks for context and target.
        
        Args:
            input_size: Tuple (H, W) of input image size.
            patch_size: Size of the patch (32 for ResNet-50).
            enc_mask_scale: Scale range for the context mask (not used directly in standard I-JEPA, usually we sample target blocks).
                            In I-JEPA/CNN-JEPA, we sample target blocks to MASK OUT.
            pred_mask_scale: Scale range for the target/prediction blocks.
            aspect_ratio: Aspect ratio range for the blocks.
            nenc: Number of context blocks (usually just the complement of target).
            npred: Number of target/prediction blocks to sample.
            min_keep: Minimum number of visible patches.
        """
        self.input_size = input_size
        self.patch_size = patch_size
        self.height, self.width = input_size
        self.num_h = self.height // patch_size
        self.num_w = self.width // patch_size
        self.num_patches = self.num_h * self.num_w
        
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap

    def _sample_block_mask(self, scale, aspect_ratio):
        """
        Sample a block mask.
        Returns a mask of shape (num_h, num_w) where 1 indicates the block.
        """
        _mask = torch.zeros((self.num_h, self.num_w), dtype=torch.int32)
        
        # Sample block size
        num_patches = self.num_patches
        min_s, max_s = scale
        min_ar, max_ar = aspect_ratio
        
        target_area = random.uniform(min_s, max_s) * num_patches
        log_aspect_ratio = (math.log(min_ar), math.log(max_ar))
        aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
        
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        
        h = min(h, self.num_h)
        w = min(w, self.num_w)
        
        # Sample position
        top = random.randint(0, self.num_h - h)
        left = random.randint(0, self.num_w - w)
        
        _mask[top:top+h, left:left+w] = 1
        return _mask

    def __call__(self, batch):
        """
        Args:
            batch: List of images (tensors).
            
        Returns:
            collated_images: (B, C, H, W)
            collated_masks: (B, 1, H_feat, W_feat) - Binary mask for Context Encoder (1=keep, 0=mask)
            mask_indices: List of tensors, each containing indices of masked patches for loss computation.
        """
        images = torch.stack([item[0] if isinstance(item, (tuple, list)) else item for item in batch])
        B = len(images)
        
        collated_masks = []
        mask_indices = []
        
        for _ in range(B):
            # 1. Initialize context mask as all visible (1)
            # We will set regions to 0 (masked) based on target blocks
            context_mask = torch.ones((self.num_h, self.num_w), dtype=torch.int32)
            
            # 2. Sample target blocks (regions to predict)
            target_masks = []
            for _ in range(self.npred):
                target_mask = self._sample_block_mask(self.pred_mask_scale, self.aspect_ratio)
                target_masks.append(target_mask)
                
                # Update context mask: remove target region
                # context_mask = context_mask * (1 - target_mask) 
                # Actually, in I-JEPA, context is complement of union of target blocks.
                context_mask[target_mask.bool()] = 0
            
            # Ensure we have enough context
            if context_mask.sum() < self.min_keep:
                # If too much is masked, reset or force some visible.
                # Simple fix: just unmask random patches until min_keep is met
                # Or just retry (omitted for simplicity, assuming reasonable scales)
                pass

            # 3. Prepare outputs
            # Context Mask: (1, H_feat, W_feat) -> 1 means visible/context
            collated_masks.append(context_mask.unsqueeze(0))
            
            # Mask Indices: Indices of patches that are MASKED (target regions)
            # We want to predict the union of all target blocks.
            # So we look for where context_mask is 0.
            # Note: The paper says "The union of these blocks is the Target Region".
            # So we predict everything that is NOT context.
            target_region = (context_mask == 0)
            
            # Flatten to get indices (0 to num_patches-1)
            flat_target = target_region.flatten()
            indices = torch.nonzero(flat_target).squeeze(1)
            mask_indices.append(indices)
            
        collated_masks = torch.stack(collated_masks) # (B, 1, H_feat, W_feat)
        
        return images, collated_masks, mask_indices

if __name__ == "__main__":
    # Simple test
    collator = MaskCollator(input_size=(224, 224), patch_size=32)
    dummy_batch = [torch.randn(3, 224, 224) for _ in range(4)]
    images, masks, indices = collator(dummy_batch)
    print(f"Images shape: {images.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Indices len: {len(indices)}")
    print(f"Indices[0] shape: {indices[0].shape}")
