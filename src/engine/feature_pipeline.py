import os
import time
from src.engine.feature_extractor import FeatureExtractor
from src.engine.wikiart_dataset import WikiArtDataset
from src.engine.upload_to_bucket import upload_to_bucket
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def skip_none_collate(batch):
    # Filter out None samples
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        # If everything is None, return None for this batch
        return None, None
    imgs, labels = zip(*batch)
    return torch.stack(imgs, 0), torch.stack(labels, 0)


class GCPFeatureExtractionPipeline:
    def __init__(self,
                 bucket_name: str,
                 architecture: str = 'resnet',
                 augment: bool = False,
                 batch_size: int = 32,
                 num_workers: int = 4):
        """
        bucket_name: 'csci-ga-2565-final-project-wikiart'
        architecture: 'alexnet', 'resnet', or 'vgg'
        augment: bool, whether to use augmentation
        """
        self.bucket_name = bucket_name
        self.architecture = architecture
        self.augment = augment
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.extractor = FeatureExtractor(architecture=self.architecture, augment=self.augment)
        self.device = self.extractor.device
        self.model = self.extractor.model

        # Create a unique log file name
        aug_suffix = "_aug" if self.augment else ""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"missing_files_{self.architecture}{aug_suffix}_{timestamp}.log"

        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        self.log_path = os.path.join('logs', self.log_filename)

    def run(self, csv_gcs_path: str, prefix: str, data_type: str):
        """
        csv_gcs_path: path to CSV in GCS (e.g. 'metadata/filtered_style_train.csv')
        prefix: 'train' or 'val' for images
        data_type: 'train' or 'valid' for output categorization
        """
        dataset = WikiArtDataset(
            csv_gcs_path=csv_gcs_path,
            bucket_name=self.bucket_name,
            prefix=prefix,
            transform=self.extractor.transform,
            log_file=self.log_path  # Pass the dynamically generated log file
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=skip_none_collate
        )

        self.model.eval()

        all_features = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="Extracting Features"):
                if imgs is None:
                    # This batch had no valid samples (all were None)
                    continue
                imgs = imgs.to(self.device, non_blocking=True)
                _ = self.model(imgs)
                feats = self.extractor.features
                feats = feats.view(feats.size(0), -1)
                all_features.append(feats.cpu())
                all_labels.append(labels.cpu())

        if len(all_features) > 0:
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            aug_suffix = "_aug" if self.augment else ""
            local_filename = f"{prefix}_{self.architecture}_features{aug_suffix}.pt"
            local_path = os.path.join("data", "processed", local_filename)

            torch.save({'features': all_features, 'labels': all_labels}, local_path)
            print(f"Saved features to {local_path}")

            # Update GCS path if desired
            gcs_path = f"{self.architecture}/{local_filename}"
            upload_to_bucket(local_path=local_path, bucket_name=self.bucket_name, gcs_path=gcs_path)
        else:
            print("No features extracted (all images were missing or invalid).")
