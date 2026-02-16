import os
from textwrap import dedent

def is_colab():
    return 'COLAB_RELEASE_TAG' in os.environ

def get_data_dir():
    """Return the appropriate data directory based on the environment."""
    if is_colab():
        return '/content/data'
    else:
        # Local: use a 'data' folder relative to the notebook/script
        #return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        # Or just use: return './data'
        return "./data"

def download_data_files(folder_id, data_dir):
    print("Downloading from Drive folder...")
    try:
        import gdown
    except ImportError:
        import subprocess
        print("Installing gdown...")
        subprocess.check_call(['pip', 'install', 'gdown'])
        import gdown

    url = f'https://drive.google.com/drive/folders/{folder_id}'
    gdown.download_folder(url, output=data_dir, quiet=False, use_cookies=False)


def ensure_data_available(folder_id, expected_files, data_dir=None):
    """Download missing files from the Google Drive folder.

    Args:
        folder_id: Google Drive folder ID
        expected_files: List of filenames expected in the folder
        data_dir: Optional path to data directory (defaults based on environment)

    Returns:
        Path to the data directory
    """
    if data_dir is None:
        data_dir = '/content/data' if is_colab() else './data'

    os.makedirs(data_dir, exist_ok=True)

    local_files = set(os.listdir(data_dir)) if os.listdir(data_dir) else set()
    missing = set(expected_files) - local_files

    if missing:
        print(f"Missing files: {missing}")
        print("Downloading from Drive folder...")

        try:
            import gdown
        except ImportError:
            import subprocess
            subprocess.check_call(['pip', 'install', 'gdown'])
            import gdown

        download_data_files(folder_id, data_dir)
        # gdown.download_folder(
        #     f'https://drive.google.com/drive/folders/{folder_id}',
        #     output=data_dir,
        #     quiet=False
        # )
    else:
        print(f"✓ All files already present in {data_dir}")

    # Verify
    local_files = set(os.listdir(data_dir))
    still_missing = set(expected_files) - local_files

    if still_missing:
        error_msg = dedent(f"""
            Failed to download all required files from Google Drive folder {folder_id}!
            - Local files: {local_files}
            - Expected files: {expected_files}
            - Failed to download: {still_missing}
        """)
        raise FileNotFoundError(error_msg)

    print(f"\nFiles available in {data_dir}:")
    for f in sorted(os.listdir(data_dir)):
        size_mb = os.path.getsize(os.path.join(data_dir, f)) / (1024 * 1024)
        print(f"  {f}: {size_mb:.1f} MB")

    return data_dir


