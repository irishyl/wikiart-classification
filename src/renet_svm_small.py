#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:48:20 2024

@author: lvxinyuan
"""

''' Notes:
style_to_keep = [12,21,23,9,20,24,3,4,0,17,15,7,22]
'''

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import io

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

''' Prep the data'''
# Load the data from Parquet
# data = pd.read_parquet("/Users/lvxinyuan/Downloads/part-00001-28e00cff-650d-419e-8569-171783f48624-c000.snappy.parquet")
data = pd.read_parquet("/Users/lvxinyuan/Downloads/train-00000-of-00072.parquet")
print(data.head())

# Get a subset of data just for testing
subData = data[:20]
# this is to remove labels with only 1 count 
label_counts = Counter(subData.iloc[:,1])
rare_labels = [label for label, count in label_counts.items() if count == 1]
subData = subData[~subData.iloc[:, 1].isin(rare_labels)]


 # Define preprocessing transformations for ResNet50
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ResNet input size
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet stats
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


# Load ResNet50
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # DEFAULT = IMAGENET1K_V1
resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1]) # feature extraction
resnet50.eval() 

# Extract Features
def extract_features(model, dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for images, labels_batch in dataloader:  # Unpack the tuple
            # Extract features
            output = model(images).squeeze()  # Output shape: [batch_size, 2048]
            features.append(output.numpy())
            labels.extend(labels_batch.numpy())  # Convert labels to numpy
    return np.vstack(features), np.array(labels)



'''main'''
# transform the data
dataset = WikiArtDataset(subData, transform=transform)

# Train-Test-Split
train_dataset, test_dataset = train_test_split(dataset, test_size=0.3, random_state=42, stratify=[item[1] for item in dataset])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
# what's inside inside the DataLoader
for images, labels in train_loader:
    print(images.shape)  # [11, 3, 224, 224]
    print(labels)        # styles
    
for images, labels in test_loader:
    print(images.shape)  # [5, 3, 224, 224]
    print(labels)        # styles


# Extract features for train and test sets
train_features, train_labels = extract_features(resnet50, train_loader)
test_features, test_labels = extract_features(resnet50, test_loader)
print("Train features shape:", train_features.shape)  # [16, 2048]
print("Test features shape:", test_features.shape)    # [4, 2048]



'''Multiclass SVM'''
# Train a multiclass SVM

# Create an SVM pipeline with feature scaling
svm = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1, decision_function_shape="ovr"))

# Train the SVM
svm.fit(train_features, train_labels)

# Predict on the test set
test_predictions = svm.predict(test_features)

# Evaluate the SVM
print("Classification Report:\n")
print(classification_report(test_labels, test_predictions))