# This file is responsible for training and eval the model in a local environment
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# Load img_label which is created by GetImageLabel
from GetImageLabel import class_to_idx
# Load Model
from Model_Configuration import CustomDataset, MyModel, ImprovedModel
import torch.multiprocessing as mp
from tqdm import tqdm


# Wrap the main code in this if statement
if __name__ == '__main__':
    # Add this line for Windows multiprocessing support
    mp.freeze_support()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Better transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Image Transform
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dataset and DataLoader
    img_dir = "D:\personal\stress\Practice_Project\Pytorch_Template\Practice_Dataset\cifar_images"  # <- Update this!
    full_dataset = CustomDataset(img_dir=img_dir, transform=train_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)



    # Training Setup
    num_classes = len(class_to_idx) if class_to_idx else 10
    model = ImprovedModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    num_epochs = 20

    # Training Loop with validation
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {running_loss/len(train_loader):.4f}, "
            f"Val Loss: {val_loss/len(val_loader):.4f}, "
            f"Val Accuracy: {val_accuracy:.2f}%")
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model saved with accuracy: {best_val_acc:.2f}%")

    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Load best model for final evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Final evaluation on validation set
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Final Accuracy on validation set: {100 * correct / total:.2f}%')
