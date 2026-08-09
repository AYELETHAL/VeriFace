
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms

# Settings
DATADIR = r"C:\Users\97258\engineering_try_2\Engineering-Project"
CATEGORIES = [
    "No_Manipulation",
    "output_faces_change_lip_color_no_padding",
    "output_faces_change_eye_color_no_padding",
    "output_faces_change_nose_no_padding",
    "output_faces_change_eyebrows_no_padding"
]

IMG_SIZE = 224


def load_data():
    data = []
    labels = []
    paths = []

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            image = cv2.imread(img_path)
            if image is None:
                continue

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            data.append(image)
            labels.append(class_num)
            paths.append(img_path)

    return np.array(data), np.array(labels), np.array(paths)

# Custom Dataset to integrate torchvision transforms
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        img_array = self.data[idx] # shape (H, W, C)
        img = Image.fromarray(img_array) # Convert to PIL Image
        
        if self.transform:
            img = self.transform(img)
            
        label = self.labels[idx]
        return img, label


# Main execution
print("=" * 60)
print("ResNet18 + Augmentation + Dropout Classification")
print("=" * 60)

X, y, paths = load_data()

if len(X) > 0:
    # 1. Split data into Train, Validation, and Test sets
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
    X, y, paths,
    test_size=0.2,
    stratify=y,
    random_state=42
)

    with open("test_paths.json", "w") as f:
        json.dump(paths_test.tolist(), f)

    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,
        stratify=y_train,
        random_state=42
    )
    
    # Define Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CustomDataset(X_train, y_train, transform=train_transform)
    val_dataset = CustomDataset(X_val, y_val, transform=val_test_transform)
    test_dataset = CustomDataset(X_test, y_test, transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 2. Define Model and Modify Classifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze the early feature extractor or unfreeze all (let's fine-tune all parameters with a low LR)
    for param in model.parameters():
        param.requires_grad = True
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5), # Add Dropout for preventing Overfitting
        nn.Linear(num_ftrs, len(CATEGORIES))
    )
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    epochs = 20
    best_val_loss = float('inf')
    
    train_losses, val_losses = [], []
    
    # 3. Train the model
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Evaluate on validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Save best model weights if Validation Loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_resnet_model.pth")
            print("  -> Model saved as best_resnet_model.pth")
            
    # Load best weights before testing
    model.load_state_dict(torch.load("best_resnet_model.pth"))
    
    # 4. Loss Plot
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 5. Evaluate on Test Set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")
    
    # 6. Confusion Matrix & Report
    short_labels = ["Original", "Lip", "Eye", "Nose", "Eyebrows"]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=short_labels))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=short_labels, 
                yticklabels=short_labels)
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()