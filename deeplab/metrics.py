import os
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from tqdm import tqdm

# Paths to folders
image_folder1 = "../../idd_datasets/IDD_w_faces/images/"
image_folder2 = "../../idd_datasets/idd_face_fake_LamaHQ/images/"

image1_files = sorted([f for f in os.listdir(image_folder1)])
image2_files = sorted([f for f in os.listdir(image_folder2)])

# Preprocessing
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the model
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
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
    image1_path = os.path.join(image_folder1, img1)
    image2_path = os.path.join(image_folder2, img2)

    input_image1 = Image.open(image1_path).convert("RGB")
    input_tensor1 = preprocess(input_image1).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor1)["out"][0]
    output1 = output.argmax(0)

    input_image2 = Image.open(image2_path).convert("RGB")
    input_tensor2 = preprocess(input_image2).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor2)["out"][0]
    output2 = output.argmax(0)

    mask1_flat = output1.view(-1).cpu().numpy()
    mask2_flat = output2.view(-1).cpu().numpy()

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