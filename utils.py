from pathlib import Path

def verify_dataset(data_path: str):
    data_path = Path(data_path)

    # Check if the main directory exists
    if not data_path.exists():
        raise ValueError(f"Dataset directory does not exist: {data_path}")

    # Check for images and labels directories
    images_path = data_path / 'images'
    labels_path = data_path / 'labels'

    if not images_path.exists():
        raise ValueError(f"Missing directory: {images_path}")
    if not labels_path.exists():
        raise ValueError(f"Missing directory: {labels_path}")

    # Verify matching images and labels
    image_files = set(f.stem for f in images_path.glob('*.jpg'))
    label_files = set(f.stem for f in labels_path.glob('*.txt'))

    if image_files != label_files:
        print(f"Warning: Mismatched images and labels in {data_path}")

    return True
