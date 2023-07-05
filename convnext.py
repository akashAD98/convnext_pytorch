

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet101
from torchvision.models import convnext_small
# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
num_epochs = 15
batch_size = 16
learning_rate = 0.001

# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
train_dataset = datasets.ImageFolder('2class/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load the validation dataset
val_dataset = datasets.ImageFolder('2class/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the model
model = convnext_small(pretrained=True)
#resnet101(pretrained=True)

#resnet101(pretrained=True)
num_features = 768
#model.fc.in_features

print("num features",num_features)
model.fc = nn.Linear(num_features, 2)  # 10 classes
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Track best model
best_loss = float('inf')
best_accuracy = 0.0

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()

        running_loss += loss.item()

        # Print training progress for each step
        print('Epoch {}/{} - Step {}/{} - Loss: {:.4f}'.format(epoch+1, num_epochs, step+1, len(train_loader), loss.item()), flush=True)

    epoch_loss = running_loss / len(train_loader)
    epoch_train_accuracy = correct_predictions / len(train_dataset)

    # Validation loop
    model.eval()
    val_correct_predictions = 0
    val_running_loss = 0.0

    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)

            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_labels)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_correct_predictions += (val_predicted == val_labels).sum().item()

            val_running_loss += val_loss.item()

    epoch_val_loss = val_running_loss / len(val_loader)
    epoch_val_accuracy = val_correct_predictions / len(val_dataset)

    # Print epoch-level metrics
    print('Epoch {}/{} - Train Loss: {:.4f} - Train Accuracy: {:.4f} - Val Loss: {:.4f} - Val Accuracy: {:.4f}'.format(epoch+1, num_epochs, epoch_loss, epoch_train_accuracy, epoch_val_loss, epoch_val_accuracy), flush=True)

    # Save the best model based on validation loss and accuracy
    if epoch_val_loss < best_loss or epoch_val_accuracy > best_accuracy:
        best_loss = epoch_val_loss
        best_accuracy = epoch_val_accuracy
        torch.save(model.state_dict(), 'BEST_New_convnext_small_20epochs.pth')

print('Training completed! Best model saved as best_model.pth.')
