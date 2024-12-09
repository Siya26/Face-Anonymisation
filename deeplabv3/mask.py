import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
# or any of these variants
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
model.eval()

input_image = Image.open("../../idd_datasets/idd_face_fake_LamaHQ/images/000216_leftImg8bit.png")
input_image = input_image.convert("RGB")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# Convert output predictions to a PIL image with the color palette applied
segmentation_image = Image.fromarray(output_predictions.byte().cpu().numpy())
segmentation_image = segmentation_image.resize(input_image.size, Image.NEAREST)
segmentation_image.putpalette(colors)

# Convert PIL image to a NumPy array (RGB)
segmentation_array = np.array(segmentation_image)

# Save the result using OpenCV
output_path = "segmentation_result_fake.png"
cv2.imwrite(output_path, cv2.cvtColor(segmentation_array, cv2.COLOR_RGB2BGR))

print(f"Segmentation result saved to {output_path}")
