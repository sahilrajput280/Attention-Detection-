import os
import torch
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import numpy as np

# Define your class names
CLASSES = ['attentive', 'not_attentive']

# Set up paths
IMAGE_FOLDER = "./DATA/test/images"
LABEL_FOLDER = "./DATA/test_cls/labels"

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class AttentionDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        label_path = os.path.join(self.label_dir, image_file.replace(".jpg", ".txt").replace(".png", ".txt"))

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        with open(label_path, 'r') as f:
            label = int(f.readline().strip())

        return image, label

# Load test data
test_ds = AttentionDataset(IMAGE_FOLDER, LABEL_FOLDER, transform)
test_loader = DataLoader(test_ds, batch_size=32)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("attention_classifier.pth", map_location=device))
model.to(device)
model.eval()

# Evaluation
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # for ROC curve (class 1 probs)

# Metrics
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASSES))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# AUC-ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.show()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy:", accuracy_score(all_labels, all_preds))
print("Precision:", precision_score(all_labels, all_preds))
print("Recall:", recall_score(all_labels, all_preds))
print("F1 Score:", f1_score(all_labels, all_preds))
