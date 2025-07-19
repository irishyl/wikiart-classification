#!/usr/bin/env python3
import sys
import os
from src.models.baseline_model_trainer import BaselineModelTrainer

def main():
    # Paths to your extracted feature files
    train_path = "data/processed/resnet_train_features.pt"
    valid_path = "data/processed/resnet_valid_features.pt"

    # Initialize the trainer for RF with shuffle_data=True
    trainer = BaselineModelTrainer(
        model_type="rf",
        output_dir="output",
        shuffle_data=True,  # Enable shuffling for training data
        n_estimators=200,
        max_depth=10
    )

    # Load datasets
    trainer.load_datasets(train_path, valid_path)

    # Initialize and train the RF
    trainer.initialize_model()
    trainer.fit()

    # Evaluate on validation set and save results
    trainer.evaluate(save_results=True)

if __name__ == "__main__":
    main()
