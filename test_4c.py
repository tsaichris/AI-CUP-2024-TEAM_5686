import torch
import os
import cv2
from PIL import Image
import numpy as np
from UNet_modified import UNet
from edge import edge_detection
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = UNet(4, 1).to(device)
state_dict = torch.load('428_2.pth', map_location=device)
model.load_state_dict(state_dict)
model.eval()

transform = A.Compose([
    A.Resize(428, 428),
    A.Normalize(mean=(0.485, 0.456, 0.406, 0.5), std=(0.229, 0.224, 0.225, 0.5)),
    ToTensorV2()
])

# input and output folders
input_folder = "C://Users//User//Desktop//drone//private"
output_folder = "C://Users//User//Desktop//drone//temp"
os.makedirs(output_folder, exist_ok=True)

image_paths = sorted(glob.glob(f"{input_folder}/*"))


with torch.no_grad():
    for image_path in image_paths:
       
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply edge detection
        edge_mask, _ = edge_detection(image)
        edge_mask = edge_mask[:, :, np.newaxis]  # Add channel dimension to edge mask
        combined_image = np.concatenate((image, edge_mask), axis=2)  # Combine image and edge mask

        # Apply the transformations
        transformed = transform(image=combined_image)
        input_tensor = transformed['image'].unsqueeze(0).to(device)

        # get the model prediction
        pred = model(input_tensor).cpu().detach()
        pred = pred > 0.5  # Apply threshold
        pred = pred.squeeze().numpy()  # Remove batch dimension and convert to numpy

        # convert prediction to a PIL image
        pred_image_pil = Image.fromarray((pred * 255).astype(np.uint8))

        # save predicition
        filename = os.path.basename(image_path)
        pred_image_pil.save(os.path.join(output_folder, filename))
