import torch
from torchvision.io import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import numpy as np
import cv2
import os
from torchvision.transforms.functional import resize

# Load the image
img = read_image("../../idd_datasets/IDD_w_faces/images/000216_leftImg8bit.png")  # Replace with your image path

# Step 1: Initialize model with the best available weights
weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model for prediction
with torch.no_grad():
    prediction = model(batch)["out"]

# Step 5: Identify categories present in the image
normalized_masks = prediction.softmax(dim=1)
categories = weights.meta["categories"]
original_height, original_width = img.shape[1:]  # Original image dimensions

# Create an empty RGB image for visualization
output_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)

# Generate unique colors for each category
np.random.seed(42)  # For reproducible color generation
colors = {cls: tuple(np.random.randint(0, 255, size=3).tolist()) for cls in categories}

for class_name in categories:
    class_idx = categories.index(class_name)
    mask = normalized_masks[0, class_idx].cpu().numpy()

    # Resize mask to original image size
    resized_mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

    # Threshold to identify significant presence of the category
    threshold = 0.5  # Adjust if needed
    if resized_mask.max() > threshold:
        print(f"Category detected: {class_name}")
        binary_mask = (resized_mask > threshold).astype(np.uint8)

        # Apply the category's color to the output image
        for c in range(3):  # For each channel (R, G, B)
            output_image[:, :, c] += (binary_mask * colors[class_name][c]).astype(np.uint8)

# Save the resulting image
output_path = "segmentation_orig.png"
cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
print(f"Combined segmentation result saved to {output_path}")
