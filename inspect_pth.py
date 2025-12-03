import torch
import sys

try:
    data = torch.load("lightvaew2_1.pth", map_location="cpu", weights_only=True)
    # Check if it's a state dict or has a 'state_dict' key
    if "state_dict" in data:
        state_dict = data["state_dict"]
    else:
        state_dict = data

    print("Keys found in checkpoint:")
    for key in sorted(state_dict.keys()):
        if "norm" in key:
            print(f"  {key}")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
