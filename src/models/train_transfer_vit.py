#!/usr/bin/env python3
import sys
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from src.engine.wikiart_dataset import WikiArtDataset
from src.models.transfer_learning import TransferLearningTrainer

def main():
    bucket_name = "csci-ga-2565-final-project-wikiart"
    train_csv = "metadata/filtered_style_train.csv"
    val_csv = "metadata/filtered_style_val.csv"
    prefix_train = "train"
    prefix_val = "val"
    num_classes = 10  # Adjust as needed

    # Transforms for ViT training (similar to ResNet)
    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = WikiArtDataset(train_csv, bucket_name, prefix_train, transform=train_transform)
    val_dataset = WikiArtDataset(val_csv, bucket_name, prefix_val, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    trainer = TransferLearningTrainer(model_type="vit", num_classes=num_classes, epochs=5, learning_rate=1e-3)
    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    main()
