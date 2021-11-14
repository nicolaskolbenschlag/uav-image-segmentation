import torch
import segmentation_models_pytorch

def load_model(device: str) -> torch.nn.Module:
    model = segmentation_models_pytorch.Unet("mobilenet_v2", encoder_weights="imagenet", classes=23, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
    model.load_state_dict(torch.load("Unet-Mobilenet.pth", map_location=device))
    model.eval()
    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"