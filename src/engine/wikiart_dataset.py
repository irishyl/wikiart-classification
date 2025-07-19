# import csv
# import os
# import io
# from torch.utils.data import Dataset
# from google.cloud import storage
# from PIL import Image
#
# class WikiArtDataset(Dataset):
#     def __init__(self, csv_gcs_path: str, bucket_name: str, prefix: str, transform=None):
#         """
#         csv_gcs_path: path to CSV in GCS (e.g. 'metadata/filtered_style_train.csv')
#         bucket_name: GCP bucket name ('csci-ga-2565-final-project-wikiart')
#         prefix: 'train' or 'val' for images
#         transform: image transform
#         """
#         self.bucket_name = bucket_name
#         self.prefix = prefix
#         self.transform = transform
#         self.samples = []
#
#         # Temporary client to download CSV only at initialization
#         tmp_client = storage.Client()
#         tmp_bucket = tmp_client.bucket(self.bucket_name)
#
#         # Download CSV from GCP
#         csv_blob = tmp_bucket.blob(csv_gcs_path)
#         csv_content = csv_blob.download_as_text()
#         reader = csv.reader(csv_content.splitlines())
#         for row in reader:
#             img_rel_path, label = row[0].strip(), int(row[1])
#             self.samples.append((img_rel_path, label))
#
#         # Do not store the client or bucket as class attributes
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         # Create a new client and bucket here to avoid pickling issues
#         client = storage.Client()
#         bucket = client.bucket(self.bucket_name)
#
#         img_rel_path, label = self.samples[idx]
#         full_path = f"{self.prefix}/{img_rel_path}"
#         blob = bucket.blob(full_path)
#         img_bytes = blob.download_as_bytes()
#
#         image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image, label
import csv
import os
import io
from torch.utils.data import Dataset
from google.cloud import storage
from google.api_core.exceptions import NotFound
from PIL import Image
import torch


class WikiArtDataset(Dataset):
    def __init__(self, csv_gcs_path: str, bucket_name: str, prefix: str, transform=None, log_file="missing_files.log"):
        """
        csv_gcs_path: Path to CSV in GCS (e.g. 'metadata/train.csv')
        bucket_name: GCP bucket name (e.g. 'csci-ga-2565-final-project-wikiart')
        prefix: 'train' or 'val' directory in the bucket
        transform: image transformations
        log_file: path to log file to record missing images
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.transform = transform
        self.samples = []
        self.log_file = log_file

        # Ensure log directory exists if log_file includes directories
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Initialize client and bucket
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)

        # Download and parse CSV
        csv_blob = bucket.blob(csv_gcs_path)
        csv_content = csv_blob.download_as_text()
        reader = csv.reader(csv_content.splitlines())

        # Assuming first column is image path and second is class label (int)
        for row in reader:
            img_rel_path = row[0].strip()
            label = int(row[1].strip())
            self.samples.append((img_rel_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)

        img_rel_path, label = self.samples[idx]
        full_path = f"{self.prefix}/{img_rel_path}"
        blob = bucket.blob(full_path)

        # Attempt to download image
        try:
            img_bytes = blob.download_as_bytes()
        except NotFound:
            # Log missing file
            with open(self.log_file, 'a') as f:
                f.write(f"{full_path}\n")
            return None

        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)

