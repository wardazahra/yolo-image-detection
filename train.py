import os
from ultralytics import YOLO
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def train_yolo(model, train_loader, val_loader, epochs, save_path):
    model.train(data=train_loader, epochs=epochs, val=val_loader, save_dir=save_path)

def main():
    # Set up your data loaders
    train_data_path = "datasets/train"  # Update with your actual train data path
    val_data_path = "datasets/val"  # Update with your actual validation data path

    train_loader = DataLoader(train_data_path, transform=ToTensor(), batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data_path, transform=ToTensor(), batch_size=8, shuffle=False, num_workers=4)

    # Set up your YOLO model
    model = YOLO("yolov8n.yaml")

    # Train the model
    save_path = "runs/train_results"  # Update with your desired save path
    epochs = 100  # Update with the desired number of epochs

    train_yolo(model, train_loader, val_loader, epochs, save_path)

if __name__ == "__main__":
    main()
