import argparse
import torch
import torchvision
import segmentation_models_pytorch
import cv2
import numpy as np
import pandas as pd
import albumentations
import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", type=str, required=False, default="video.mp4")
    parser.add_argument("--show", type=bool, default=True)

    args = parser.parse_args()
    return args

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

COLORS = {
    5: [0,255,0]
}

def render(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = segmentation_models_pytorch.Unet("mobilenet_v2", encoder_weights="imagenet", classes=23, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
    model.load_state_dict(torch.load("Unet-Mobilenet.pth", map_location=device))
    model.eval()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        pass

    t = albumentations.Resize(768, 1152, interpolation=cv2.INTER_NEAREST)
    colors = pd.read_csv("class_dict_seg.csv")
    
    out = cv2.VideoWriter(f"{args.video.split('.')[0]}_out.avi" , cv2.VideoWriter_fourcc(*"MJPG"), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    progress_bar = tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            t_mask = albumentations.Resize(frame.shape[0], frame.shape[1], interpolation=cv2.INTER_NEAREST)

            frame_resized = t(image=frame)["image"]

            if count % 100 == 0:
                mask = predict_mask(model, t(image=cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))["image"], device)
                mask = t_mask(image=mask)["image"]
            
            frame_mask = np.zeros_like(frame)
            for idx in range(len(colors)):
                name = colors["name"][idx]
                if not name in ["paved-area", "dirt", "grass", ""]:
                    continue
                color = (colors["r"][idx], colors["g"][idx], colors["b"][idx])
                frame_mask = np.where((mask == idx)[...,None], np.array(color, dtype="uint8"), frame_mask)
            
            annotated_frame = cv2.addWeighted(frame, .5, frame_mask, .5, 0)#.8,.2
            out.write(annotated_frame)

            count += 1
            progress_bar.update(count)
            
            if args.show:
                cv2.imshow(f"FlyAI: {args.video}", annotated_frame)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

        else:
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    render(parse_args())