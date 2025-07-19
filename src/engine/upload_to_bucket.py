from google.cloud import storage

def upload_to_bucket(local_path: str, bucket_name: str, gcs_path: str):
    """
    Upload a local file to a specified Google Cloud Storage path.

    Parameters
    ----------
    local_path : str
        The path to the local file you want to upload.
    bucket_name : str
        The name of the GCP bucket.
    gcs_path : str
        The desired path (including filename) within the bucket.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")
