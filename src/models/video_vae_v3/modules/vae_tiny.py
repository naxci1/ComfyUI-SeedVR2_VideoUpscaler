import torch
import torch.nn as nn

from .tae import TAEHV

class WanVAE_tiny(nn.Module):
    """
    Wrapper for Tiny AutoEncoder (TAEHV) configured for Wan2.1.
    """
    def __init__(self, vae_path="taew2_1.pth", dtype=torch.bfloat16, device="cuda", need_scaled=True):
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
            self.register_buffer('shift', torch.tensor(self.latents_mean).view(1, self.z_dim, 1, 1, 1))
            self.register_buffer('scale', 1.0 / torch.tensor(self.latents_std).view(1, self.z_dim, 1, 1, 1))

    @torch.no_grad()
    def decode(self, latents, return_dict=False):
        # SeedVR2 interface: encode/decode take/return tensors or objects
        # latents: [B, C, T, H, W]

        # Ensure latents are on correct device/dtype
        latents = latents.to(self.dtype)

        if self.need_scaled:
            # Apply scaling (denormalization for decoding)
            # z = (z - mean) * scale_inv  <-- wait, standard VAE is: z = (x - mean) * scale_inv
            # decode: x = z / scale_inv + mean = z * scale + mean

            # FlashVSR code: latents = latents / latents_std + latents_mean
            # My buffer 'scale' is 1.0 / latents_std
            # So: latents = latents * self.scale + self.shift

            # Ensure buffers are on correct device
            if self.shift.device != latents.device:
                self.shift = self.shift.to(latents.device).to(latents.dtype)
                self.scale = self.scale.to(latents.device).to(latents.dtype)

            latents = latents * self.scale + self.shift

        # Decode
        # Transpose input to [B, C, T, H, W] -> internally TAEHV expects NTCHW?
        # TAEHV.decode_video expects NTCHW RGB (C=3)
        # Wait, decode_video input args: x: input NTCHW latent (C=12)
        # But Wan2.1 has 16 channels. TAEHV init set latent_channels=16.

        # FlashVSR VAE_tiny decode:
        # return self.taehv.decode_video(latents.transpose(1, 2).to(self.dtype), parallel=False).transpose(1, 2).mul_(2).sub_(1)

        # Note: SeedVR2 likely works with [B, C, T, H, W].
        # If TAEHV expects [N, C, T, H, W], it's the same.
        # But FlashVSR does transpose(1, 2)?
        # Latents shape [B, C, T, H, W]
        # Transpose(1, 2) -> [B, T, C, H, W] ?
        # Let's check TAEHV.decode_video again.
        # "TAESDV operates on NTCHW tensors"
        # N=Batch, T=Time.
        # If input is [B, C, T, H, W], we need [B, T, C, H, W]?
        # FlashVSR: latents.transpose(1, 2) -> swaps C and T.
        # So input to decode_video is [B, T, C, H, W].

        # My TAEHV implementation:
        # def decode_video(self, x, parallel=True, show_progress_bar=True):
        #    N, T, C, H, W = x.shape
        # So yes, it expects [N, T, C, H, W].

        # So we must transpose.

        decoded = self.taehv.decode_video(latents.transpose(1, 2), parallel=False, show_progress_bar=False)

        # Output is [N, T, C, H, W]. Transpose back to [N, C, T, H, W]
        decoded = decoded.transpose(1, 2)

        # FlashVSR does .mul_(2).sub_(1) -> [0, 1] to [-1, 1]
        # SeedVR2 usually expects decoded images in [0, 1] (or whatever the pipeline expects).
        # Standard VAE decode usually returns [-1, 1] or [0, 1].
        # SeedVR2 LightVAE decode returns [0, 1].
        # LightVAE: y = y.mul_(0.5).add_(0.5).clamp_(0, 1)  (if original was [-1, 1])
        # TAEHV output is [0, 1] (sigmoid/clamp at end).
        # FlashVSR converts it to [-1, 1].
        # I should check what SeedVR2 expects.
        # LightVAE code:
        # def decode_video(self, x, scale=[0, 1]):
        #    y = self.decode(y, scale).clamp_(-1, 1)
        #    y = y.mul_(0.5).add_(0.5).clamp_(0, 1)
        # So LightVAE.decode returns [-1, 1].

        # If I want to match LightVAE interface, I should probably return [-1, 1].
        # Then the pipeline converts it.
        # BUT, if I am wrapping it, maybe I should return what `decode` returns.

        # FlashVSR returns: mul(2).sub(1) -> 0*2-1=-1, 1*2-1=1. So [-1, 1].

        decoded = decoded.mul(2).sub(1)

        if return_dict:
             # Mimic diffusers DecoderOutput
             from diffusers.models.modeling_outputs import DecoderOutput
             return DecoderOutput(sample=decoded)
        return decoded

    @torch.no_grad()
    def encode(self, x, return_dict=False):
        # x: [B, C, T, H, W] (video frames)
        x = x.to(self.dtype)

        # Check input range. SeedVR2 usually passes [-1, 1] or [0, 1]?
        # LightVAE encode: y = x.mul(2).sub_(1) -> assumes x is [0, 1] and converts to [-1, 1]?
        # Wait, LightVAE.encode_video does that. LightVAE.encode expects normalized?
        # Let's assume input x is in [-1, 1].
        # TAEHV expects [0, 1].
        # So x = x * 0.5 + 0.5

        x_in = x * 0.5 + 0.5

        # Transpose to [B, T, C, H, W]
        x_in = x_in.transpose(1, 2)

        encoded = self.taehv.encode_video(x_in, parallel=False, show_progress_bar=False)

        # Transpose back to [B, C, T, H, W]
        encoded = encoded.transpose(1, 2)

        if self.need_scaled:
            # z = (z - mean) * scale_inv
            # self.scale is 1/std (scale_inv)
            # encoded = (encoded - shift) * scale

            if self.shift.device != encoded.device:
                self.shift = self.shift.to(encoded.device).to(encoded.dtype)
                self.scale = self.scale.to(encoded.device).to(encoded.dtype)

            encoded = (encoded - self.shift) * self.scale

        if return_dict:
             # Mimic diffusers AutoencoderKLOutput
             # But we return a distribution-like object
             from diffusers.models.modeling_outputs import AutoencoderKLOutput
             class PseudoDist:
                def __init__(self, val): self.val = val
                def sample(self): return self.val
                def mode(self): return self.val
             return AutoencoderKLOutput(latent_dist=PseudoDist(encoded))

        return encoded

    def set_causal_slicing(self, *args, **kwargs):
        pass

    def set_memory_limit(self, *args, **kwargs):
        pass
