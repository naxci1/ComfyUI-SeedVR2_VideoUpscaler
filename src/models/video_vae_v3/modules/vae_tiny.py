import torch
import torch.nn as nn

from .tae import TAEHV

class WanVAE_tiny(nn.Module):
    """
    Wrapper for Tiny AutoEncoder (TAEHV) configured for Wan2.1.
    """
    def __init__(self, vae_path="taew2_1.pth", dtype=torch.bfloat16, device="cuda", need_scaled=True, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device

        # Initialize TAEHV (weights loaded inside if path exists, otherwise user must load)
        # Note: TAEHV implementation handles loading if path is provided
        self.taehv = TAEHV(vae_path, latent_channels=16).to(self.dtype)

        self.temperal_downsample = [True, True, False]
        self.need_scaled = need_scaled

        if self.need_scaled:
            # Mean and Std for Wan2.1 Latents
            self.latents_mean = [
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
            ]

            self.latents_std = [
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
            ]

            self.z_dim = 16

            # Register as buffers so they move with model
            # Explicitly use device='cpu' to ensure they are materialized even in meta context
            self.register_buffer('shift', torch.tensor(self.latents_mean, device='cpu').view(1, self.z_dim, 1, 1, 1))
            self.register_buffer('scale', 1.0 / torch.tensor(self.latents_std, device='cpu').view(1, self.z_dim, 1, 1, 1))

    class TinyVAEOutput:
        def __init__(self, output):
            self.latent = output
            self.sample = output # For decode
            # For posterior.mode()
            self.posterior = self

        def mode(self):
            return self.latent

    @torch.no_grad()
    def decode(self, latents, return_dict=True, tiled=False, **kwargs):
        # SeedVR2 interface: encode/decode take/return tensors or objects
        # latents: [B, C, T, H, W]

        # Ensure latents are on correct device/dtype
        latents = latents.to(self.dtype)

        if self.need_scaled:
            # Ensure buffers are on correct device and dtype
            if self.shift.device != latents.device or self.shift.dtype != latents.dtype:
                self.shift = self.shift.to(device=latents.device, dtype=latents.dtype)
                self.scale = self.scale.to(device=latents.device, dtype=latents.dtype)

            # Denormalize: z = z_norm * std + mean
            # self.scale stores 1/std. So z = z_norm / self.scale + self.shift
            latents = latents / self.scale + self.shift

        # Decode
        # TAEHV expects NTCHW. Transpose BCTHW -> BTCHW
        # FlashVSR approach: transpose(1, 2) -> B, T, C, H, W
        decoded = self.taehv.decode_video(latents.transpose(1, 2), parallel=False, show_progress_bar=False)

        # Output is [N, T, C, H, W]. Transpose back to [N, C, T, H, W]
        decoded = decoded.transpose(1, 2)

        # FlashVSR returns [-1, 1] range
        decoded = decoded.mul(2).sub(1)

        if return_dict:
             return self.TinyVAEOutput(decoded)
        return decoded

    @torch.no_grad()
    def encode(self, x, return_dict=True, tiled=False, **kwargs):
        # x: [B, C, T, H, W] (video frames)
        x = x.to(self.dtype)

        # Input assumed to be [-1, 1]. TAEHV expects [0, 1].
        x_in = x * 0.5 + 0.5

        # Transpose to [B, T, C, H, W]
        x_in = x_in.transpose(1, 2)

        encoded = self.taehv.encode_video(x_in, parallel=False, show_progress_bar=False)

        # Transpose back to [B, C, T, H, W]
        encoded = encoded.transpose(1, 2)

        if self.need_scaled:
            if self.shift.device != encoded.device or self.shift.dtype != encoded.dtype:
                self.shift = self.shift.to(device=encoded.device, dtype=encoded.dtype)
                self.scale = self.scale.to(device=encoded.device, dtype=encoded.dtype)

            encoded = (encoded - self.shift) * self.scale

        if return_dict:
             return self.TinyVAEOutput(encoded)

        return encoded

    def set_causal_slicing(self, *args, **kwargs):
        pass

    def set_memory_limit(self, *args, **kwargs):
        pass
