import torch
from torchvision.io import read_image
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights
import numpy as np
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score, accuracy_score

# Load the image
image_folder1 = "../../idd_datasets/IDD_w_faces/images/"
image_folder2 = "../../idd_datasets/idd_face_fake_LamaHQ/images/"

image1_files = sorted([f for f in os.listdir(image_folder1)])
image2_files = sorted([f for f in os.listdir(image_folder2)])

# Step 1: Initialize model with the best available weights
weights = FCN_ResNet101_Weights.DEFAULT
model = fcn_resnet101(weights=weights)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def dice_coefficient(mask1_flat, mask2_flat):
    """
    Calculate Dice Coefficient between two flattened masks.
    """
    intersection = np.sum(mask1_flat == mask2_flat)
    return (2 * intersection) / (len(mask1_flat) + len(mask2_flat))

all_dice = []  # To store Dice Coefficient for each pair
all_miou = []  # To store mIoU for each pair
all_precision = []  # To store Precision for each pair
all_recall = []  # To store Recall for each pair
all_f1 = []  # To store F1-score for each pair
all_pixel_accuracy = []  # To store Pixel Accuracy for each pair

for img1, img2 in tqdm(zip(image1_files, image2_files), total=len(image1_files)):    
# Step 2: Initialize the inference transforms
    image1_path = os.path.join(image_folder1, img1)
    image2_path = os.path.join(image_folder2, img2)

    preprocess = weights.transforms()

    img1 = read_image(image1_path)
    batch = preprocess(img1).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(batch)["out"]
    output1 = prediction.argmax(1).squeeze(0).cpu().numpy()

    #2
    img2 = read_image(image2_path)
    batch = preprocess(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(batch)["out"]
    output2 = prediction.argmax(1).squeeze(0).cpu().numpy()

    mask1_flat = output1.flatten()
    mask2_flat = output2.flatten()

    miou = jaccard_score(mask1_flat, mask2_flat, average='macro')  # IoU
    precision = precision_score(mask1_flat, mask2_flat, average='macro', zero_division=0)
    recall = recall_score(mask1_flat, mask2_flat, average='macro', zero_division=0)
    f1 = f1_score(mask1_flat, mask2_flat, average='macro', zero_division=0)
    pixel_acc = accuracy_score(mask1_flat, mask2_flat)
    dice = dice_coefficient(mask1_flat, mask2_flat)  # Dice Coefficient

    # Store results
    all_miou.append(miou)
    all_precision.append(precision)
    all_recall.append(recall)
    all_f1.append(f1)
    all_pixel_accuracy.append(pixel_acc)
    all_dice.append(dice)

# Average metrics
average_miou = np.mean(all_miou)
average_precision = np.mean(all_precision)
average_recall = np.mean(all_recall)
average_f1 = np.mean(all_f1)
average_pixel_accuracy = np.mean(all_pixel_accuracy)
average_dice = np.mean(all_dice)

print(f"Average mIoU: {average_miou:.4f}")
print(f"Average Pixel Accuracy: {average_pixel_accuracy:.4f}")
print(f"Average Precision: {average_precision:.4f}")
print(f"Average Recall: {average_recall:.4f}")
print(f"Average F1-score: {average_f1:.4f}")
print(f"Average Dice Coefficient: {average_dice:.4f}")    