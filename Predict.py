import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

model_path = "path to trained model"
img_height, img_width = 224, 224
class_labels = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold',
                'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite',
                'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus',
                'Tomato_mosaic_virus', 'healthy', 'powdery_mildew']

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
])

# Load model
model = models.resnet50(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.5),
    nn.Linear(512, len(class_labels)),
    nn.Softmax(dim=1)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Prediction function
def predict_image(img_path):
    image_pil = Image.open(img_path).convert("RGB")
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = outputs.cpu().numpy()[0]
        top_pred = np.argmax(probs)
        top_label = class_labels[top_pred]
        confidence = probs[top_pred]

    # Plotting image with prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(image_pil)
    plt.title(f"Predicted: {top_label} ({confidence*100:.2f}%)")
    plt.axis("off")
    plt.show()

    # Plotting all class probabilities
    plt.figure(figsize=(10, 4))
    plt.bar(class_labels, probs, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Confidence")
    plt.title("Prediction Probabilities")
    plt.tight_layout()
    plt.show()

    return top_label, confidence, probs

# Example usage
image_paths = [
   "Path to Image",
   "Path to Image"
]

for img in image_paths:
    if os.path.exists(img):
        label, conf, all_probs = predict_image(img)
        print(f"✅ Predicted: {label} ({conf*100:.2f}%)")
    else:
        print(f"❌ Image not found: {img}")
