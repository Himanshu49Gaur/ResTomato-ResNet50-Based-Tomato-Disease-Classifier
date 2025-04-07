import torch
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
import os

# Config
model_path = "path to trained model"
data_dir = "path to preprocessed data"
num_samples = 3
class_labels = ['class0', 'class1', 'class2', 'class3', 'class4',
                'class5', 'class6', 'class7', 'class8', 'class9', 'class10']  # Update with actual labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformation (no augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)
_, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

# For visualization
val_dataset_vis = datasets.ImageFolder(data_dir, transform=transforms.Compose([
    transforms.Resize((224, 224)),
]))

# Load model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(512),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, 11),
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# LIME explainer
explainer = lime_image.LimeImageExplainer()

# Prediction function for LIME
def batch_predict(images):
    model.eval()
    batch = torch.stack([transform(Image.fromarray(img)) for img in images], dim=0).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()

# Visualize LIME
for i in range(num_samples):
    img_path, _ = val_dataset_vis.samples[i]
    image = Image.open(img_path).convert('RGB')
    np_img = np.array(image)

    explanation = explainer.explain_instance(
        np_img,
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(mark_boundaries(temp, mask))
    plt.title(f"LIME Explanation - Predicted: {class_labels[top_label]}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
