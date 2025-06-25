import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

st.set_page_config(page_title="FER Trainer", layout="centered")
st.title("🧠 FER Model Trainer (GPU/CPU Compatible)")

# Input fields
train_dir = st.text_input("Training Data Directory", r"dataset\train")
val_dir = st.text_input("Validation Data Directory", r"dataset\val")
model_path = st.text_input("Model Save Path", r"models\fer_model.h5")
log_dir = st.text_input("TensorBoard Log Directory", r"logs")

device_option = st.radio("Select Device", ["GPU (0)", "CPU (-1)"])
device = torch.device("cuda:0" if "GPU" in device_option and torch.cuda.is_available() else "cpu")

epochs = st.number_input("Epochs", 1, 100, 10)
batch_size = st.number_input("Batch Size", 1, 128, 32)

# Define the CNN model
class FERModel(nn.Module):
    def __init__(self, num_classes=4):
        super(FERModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),  # Adjust based on input size (224x224)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

if st.button("🚀 Start Training"):
    st.write(f"🖥 Using device: `{device}`")

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    st.write("🔄 Loading datasets...")
    try:
        train_ds = ImageFolder(train_dir, transform=transform)
        val_ds = ImageFolder(val_dir, transform=transform)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        st.success("✅ Datasets Ready")
        st.write("📦 Building the model...")

        # Initialize model, loss, and optimizer
        model = FERModel(num_classes=4).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        st.write("🏋️ Training started...")
        with st.spinner("Training..."):
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                train_loss = running_loss / len(train_loader)
                train_acc = 100 * correct / total

                # Validation
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                val_loss = val_loss / len(val_loader)
                val_acc = 100 * correct / total

                st.write(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save the model
        torch.save(model.state_dict(), model_path)
        st.success(f"✅ Model saved to `{model_path}`")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")