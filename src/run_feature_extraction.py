#!/usr/bin/env python3
import argparse
from src.engine.feature_pipeline import GCPFeatureExtractionPipeline

def main():
    parser = argparse.ArgumentParser(description="Run feature extraction with given architecture and augmentation.")
    parser.add_argument("--architecture", type=str, default="resnet",
                        choices=["resnet", "alexnet", "vgg"],
                        help="Model architecture to use.")
    parser.add_argument("--augment", action="store_true",
                        help="If set, use data augmentation during feature extraction.")
    args = parser.parse_args()

    # Construct suffix for filenames if augmentation is enabled
    aug_suffix = "_aug" if args.augment else ""

    # Example CSV paths (adjust as necessary)
    train_csv = "metadata/filtered_style_train.csv"
    val_csv = "metadata/filtered_style_val.csv"

    # Initialize pipeline with given architecture and augmentation
    pipeline = GCPFeatureExtractionPipeline(
        bucket_name="csci-ga-2565-final-project-wikiart",
        architecture=args.architecture,
        augment=args.augment,
        batch_size=64,
        num_workers=8
    )

    # Extract features for training set
    pipeline.run(
        csv_gcs_path=train_csv,
        prefix="train",
        data_type="train"
    )

    # Extract features for validation set
    pipeline.run(
        csv_gcs_path=val_csv,
        prefix="val",
        data_type="valid"
    )

    # After running this, you should get:
    # train_resnet_features.pt, val_resnet_features.pt (no augmentation, resnet)
    # train_resnet_features_aug.pt, val_resnet_features_aug.pt (augmentation, resnet)
    # ... and similarly for alexnet and vgg if you run with those arguments.

if __name__ == "__main__":
    main()
