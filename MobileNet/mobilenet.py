import torch
import torch.nn as nn
import torch.optim as optim
import os.path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust color properties
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for RGB images
])

cwd=os.getcwd()

path=os.path.join(cwd, '..', 'Original Data', 'Original_100')

dataset=ImageFolder(root=path, transform=transform)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.mobilenet_v2(weights=None)

model.features[0][0].in_channels = 3
model.classifier[1] = nn.Linear(in_features=1280, out_features=3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# model.load_state_dict(torch.load("mobilenet_75.pth"))
# model.eval()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Learning rate scheduler (optional)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

num_epochs = 50

for epoch in range(num_epochs):
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

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)


    # Compute metrics
    train_loss = running_loss / total
    train_acc = 100. * correct / total


    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, ")

    scheduler.step()  # Adjust learning rate

    # Save Model
    torch.save(model.state_dict(), "mobilenet_100.pth")

print("Training complete")

print("\nTesting the model after training...")

model.eval()
test_loss = 0.0
test_corrects = 0
test_total = 0
test_precision = 0.0
test_recall = 0.0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        test_corrects += (preds == labels).sum().item()
        test_total += labels.size(0)

# Compute test averages
test_accuracy = test_corrects / test_total * 100
avg_test_loss = test_loss / len(test_loader)


# Print test results

print(f"Test Accuracy: {test_accuracy:.2f}% | Test Loss: {avg_test_loss:.4f}")
