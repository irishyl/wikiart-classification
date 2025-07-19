import os
import pandas as pd
from google.cloud import storage
from zipfile import ZipFile
import argparse
from concurrent.futures import ThreadPoolExecutor


def create_output_directory(output_dir):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def authenticate_with_gcp():
    """Ensure Google Cloud authentication is properly set up."""
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        print("Error: GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
        print("Set the variable to point to your service account JSON key file.")
        exit(1)
    print("GCP authentication is set up properly.")

def extract_images_from_zip(zip_file, file_list, temp_dir):
    """
    Extract specific images from a ZIP file to a temporary directory.
    Args:
        zip_file (str): Path to the ZIP file.
        file_list (list): List of file paths to extract from the ZIP file.
        temp_dir (str): Temporary directory to store the extracted files.
    Returns:
        list: List of extracted file paths.
    """
    extracted_files = []

    with ZipFile(zip_file, 'r') as zf:
        zip_contents = [name for name in zf.namelist() if not name.endswith('/')]
        print(f"ZIP contains {len(zip_contents)} files after filtering directories. First 5 entries: {zip_contents[:5]}")  # Debug

        for file_name in file_list:
            if file_name in zip_contents:
                try:
                    zf.extract(file_name, temp_dir)
                    extracted_files.append(os.path.join(temp_dir, file_name))
                except Exception as e:
                    print(f"Failed to extract {file_name}: {e}")  # Log the error and continue
            else:
                print(f"File {file_name} not found in ZIP.")  # Debug: Unmatched files

    print(f"Extracted {len(extracted_files)} images to {temp_dir}.")
    return extracted_files


def upload_to_gcp_bucket(bucket_name, files, prefix, checkpoint_file):
    """
    Upload files to a GCP bucket under the specified prefix (train/ or val/).
    Args:
        bucket_name (str): Name of the GCP bucket.
        files (list): List of file paths to upload.
        prefix (str): Prefix for the bucket path (e.g., 'train/' or 'val/').
        checkpoint_file (str): Path to the checkpoint file.
    """
    # Load the checkpoint
    uploaded_files = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            uploaded_files = set(line.strip() for line in f.readlines())

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    def upload_file(file_path):
        # Generate the destination path
        local_style_path = os.path.dirname(file_path).split(os.sep)[-1]
        file_name = os.path.basename(file_path)
        blob_name = f"{prefix}{local_style_path}/{file_name}"

        if blob_name in uploaded_files:
            print(f"Skipping already uploaded file: {blob_name}")
            return

        # Upload the file
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
        print(f"Uploaded {file_path} to gs://{bucket_name}/{blob_name}")

        # Add to checkpoint
        with open(checkpoint_file, 'a') as f:
            f.write(f"{blob_name}\n")

    # Use ThreadPoolExecutor for parallel uploads
    with ThreadPoolExecutor() as executor:
        executor.map(upload_file, files)


def process_and_upload(zip_file, csv_file, temp_dir, bucket_name, prefix, checkpoint_file):
    """
    Process a dataset, extract images from the ZIP file, and upload them to the GCP bucket.
    Args:
        zip_file (str): Path to the ZIP file containing images.
        csv_file (str): Path to the CSV file (filtered_style_train.csv or filtered_style_val.csv).
        temp_dir (str): Temporary directory to extract images.
        bucket_name (str): Name of the GCP bucket.
        prefix (str): Prefix for the bucket path (e.g., 'train/' or 'val/').
        checkpoint_file (str): Path to the checkpoint file.
    """
    # Load the filtered CSV
    df = pd.read_csv(csv_file, header=None, names=["filename", "class_id"])

    # Ensure file paths include the 'wikiart/' prefix
    file_list = df['filename'].apply(lambda x: x if x.startswith("wikiart/") else f"wikiart/{x}").tolist()

    # Extract images from the ZIP file
    extracted_files = extract_images_from_zip(zip_file, file_list, temp_dir)

    # Upload images to the GCP bucket
    upload_to_gcp_bucket(bucket_name, extracted_files, prefix, checkpoint_file)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Upload top 10 class images to GCP bucket.")
    parser.add_argument(
        "--zip_file", default="../../utilities/data_ingestion/wikiart.zip",
        help="Path to the ZIP file containing all images."
    )
    parser.add_argument(
        "--train_csv", default="../../data/processed/metadata/filtered_style_train.csv",
        help="Path to the filtered_style_train.csv file."
    )
    parser.add_argument(
        "--val_csv", default="../../data/processed/metadata/filtered_style_val.csv",
        help="Path to the filtered_style_val.csv file."
    )
    parser.add_argument(
        "--temp_dir", default="../../data/temp",
        help="Temporary directory to extract images."
    )
    parser.add_argument(
        "--bucket_name", default="csci-ga-2565-final-project-wikiart",
        help="Name of the GCP bucket."
    )
    parser.add_argument(
        "--checkpoint_dir", default="../../data/checkpoints",
        help="Directory to store checkpoint files."
    )
    args = parser.parse_args()

    # Ensure Google Cloud authentication is properly set up
    authenticate_with_gcp()

    # Ensure the temporary directory exists
    create_output_directory(args.temp_dir)

    # Ensure the checkpoint directory exists
    create_output_directory(args.checkpoint_dir)

    # Process and upload training data
    print("\nProcessing training data...")
    process_and_upload(
        args.zip_file,
        args.train_csv,
        args.temp_dir,
        args.bucket_name,
        "train/",
        os.path.join(args.checkpoint_dir, "train_checkpoint.txt")
    )

    # Process and upload validation data
    print("\nProcessing validation data...")
    process_and_upload(
        args.zip_file,
        args.val_csv,
        args.temp_dir,
        args.bucket_name,
        "val/",
        os.path.join(args.checkpoint_dir, "val_checkpoint.txt")
    )

    # Cleanup: Optionally remove temporary directory
    print("\nUpload completed. Clean up the temporary directory if needed.")
