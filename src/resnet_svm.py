#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:00:58 2024

@author: lvxinyuan
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import io
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

'''prep'''

# Define transformations for training and validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize images to resnet50 input size
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # matching parameters of the pretrained weights (ImageNet)
                         std=[0.229, 0.224, 0.225])  
    
])

# Custom Dataset Class
class WikiArtDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): Dataframe containing the image and labels.
            transform: Transform to apply to the images.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Extract the image bytes and decode
        image_data = row['image']['bytes']
        image = Image.open(io.BytesIO(image_data))
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Extract the label
        label = row['style']  
        
        return image, label

# Function to process and extract features from a single Parquet file
def process_parquet(file_path, model, transform, batch_size=32):
    # Load Parquet file
    data = pd.read_parquet(file_path)
    dataset = WikiArtDataset(data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # (divides dataset into batches)

    features = []
    labels = []

    # Feature extraction using ResNet50
    with torch.no_grad():
        for images, labels_batch in dataloader:
            output = model(images).squeeze()  # Extract features
            features.append(output.numpy())
            labels.extend(labels_batch.numpy())  # Convert to numpy

    return np.vstack(features), np.array(labels)


# Load ResNet50
## stated in our proposal
## downside: trainied on natural images, might not perform that well -> Vision Transformers (ViT) may be better
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # DEFAULT = IMAGENET1K_V1
resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1]) # feature extraction
resnet50.eval() # evaluation mode, ensure consistent feature embeddings (disable dropout and batch normalization updates)

'''main'''

# Directory containing Parquet files
parquet_dir = "/scratch/hl5679/wikiart/data_cleaned"

# Iterate over Parquet files and extract features
all_features = []
all_labels = []

for file_name in os.listdir(parquet_dir):
    if file_name.endswith(".parquet"):
        file_path = os.path.join(parquet_dir, file_name)
        print(f"Processing {file_path}...")
        features, labels = process_parquet(file_path, resnet50, transform)
        all_features.append(features)
        all_labels.append(labels)

# Combine all features and labels
all_features = np.vstack(all_features)
all_labels = np.concatenate(all_labels)

# Train/Test Split
train_features, test_features, train_labels, test_labels = train_test_split(
    all_features, all_labels, test_size=0.3, random_state=42, stratify=all_labels
)

print("Train features shape:", train_features.shape)  # [num_samples, 2048]
print("Test features shape:", test_features.shape)    # [num_samples, 2048]



# Train a multiclass SVM
# Create an SVM pipeline with feature scaling
svm = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1, decision_function_shape="ovr"))
svm.fit(train_features, train_labels)
test_predictions = svm.predict(test_features)


# Evaluate the SVM
print("Classification Report:\n")
print(classification_report(test_labels, test_predictions))

