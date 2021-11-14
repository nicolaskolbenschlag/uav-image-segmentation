import torch
import segmentation_models_pytorch
import torchvision
import albumentations
import numpy as np
import cv2

def load_model(device: str) -> torch.nn.Module:
    model = segmentation_models_pytorch.Unet("mobilenet_v2", encoder_weights="imagenet", classes=23, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
    model.load_state_dict(torch.load("src/model/Unet-Mobilenet.pth", map_location=device))
    model.eval()
    return model

@torch.no_grad()
def predict_mask(model: torch.nn.Module, image: np.ndarray, device: str) -> np.ndarray:
    mean, std  = [.485, .456, .406], [.229, .224, .225]
    t = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])
    image = t(image)
    image = image.to(device)
    
    image = image.unsqueeze(0)
    output = model(image)
    masked = torch.argmax(output, dim=1)
    masked = masked.cpu().squeeze(0)

    return masked.numpy()

# NOTE for real usage, we might want to reuse the transforms as done in render_video.py
def infer_frame(model: torch.nn.Module, frame: np.ndarray, device: str) -> np.ndarray:
    t = albumentations.Resize(768, 1152, interpolation=cv2.INTER_NEAREST)
    t_mask = albumentations.Resize(frame.shape[0], frame.shape[1], interpolation=cv2.INTER_NEAREST)

    mask = predict_mask(model, t(image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))["image"], device)
    mask = t_mask(image=mask)["image"]
    return mask