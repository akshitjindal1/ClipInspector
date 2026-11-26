import torch


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class AdditiveTriggerApplier:
    """Apply a saved margin reconstruction to normalized CLIP images."""

    def __init__(self, delta, image_size):
        size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        if delta.ndim == 3:
            delta = delta.unsqueeze(0)
        if delta.shape != (1, 3, *size):
            raise ValueError(f"Unexpected shape {delta.shape} for a reconstruction at {size}")
        self.delta = delta.detach()

    def to(self, device, dtype):
        self.delta = self.delta.to(device=device, dtype=dtype)
        mean = self.delta.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = self.delta.new_tensor(CLIP_STD).view(1, 3, 1, 1)
        self.lower_bound = -mean / std
        self.upper_bound = (1 - mean) / std
        return self

    def apply(self, images):
        return torch.clamp(images + self.delta, self.lower_bound, self.upper_bound)
