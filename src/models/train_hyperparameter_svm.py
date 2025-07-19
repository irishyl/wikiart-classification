#!/usr/bin/env python3
import sys
import os
# from src.models.baseline_model_trainer import BaselineModelTrainer
from baseline_model_trainer import BaselineModelTrainer

def main():
    # Paths to your extracted feature files
    # train_path = "data/processed/resnet_train_features.pt"
    # valid_path = "data/processed/resnet_valid_features.pt"
    train_path = "/Users/lvxinyuan/me/1-Projects/NYU/1-Courses/24_Fall_ML/train_resnet_features.pt"
    valid_path = "/Users/lvxinyuan/me/1-Projects/NYU/1-Courses/24_Fall_ML/val_resnet_features.pt"

    # Initialize the trainer for SVM with shuffle_data=True
    trainer = BaselineModelTrainer(
        model_type="svm",
        output_dir="output",
        shuffle_data=True,  # Enable shuffling for training data
        kernel='rbf',
        C=1.0
    )

    # Load datasets
    trainer.load_datasets(train_path, valid_path)

    # Scaling data (included just in case since Resnet already scaled the data)
    trainer.preprocess_data()

    # Hyperparameter tuning for SVM
    param_grid = {
        'C': [0.1, 1, 10, 100],              # Regularization strength
        'kernel': ['linear', 'rbf'],         # Kernel types
        # 'gamma': ['scale', 'auto']           # Kernel coefficient for RBF
    }
    trainer.tune_hyperparameters(param_grid, search_type="random", cv=2, scoring="accuracy")

    # Initialize and train the SVM with the best hyperparameters
    trainer.initialize_model()

    # Fit the model with best hyperparameters
    trainer.fit()

    # Evaluate on validation set and save results
    trainer.evaluate(save_results=True)

if __name__ == "__main__":
    main()
