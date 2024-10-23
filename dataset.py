from pathlib import Path
import cv2
from torch.utils.data import Dataset
from collections import Counter

class PPEDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.image_files = list(self.img_dir.glob('*.jpg'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                labels = [line.strip().split() for line in f]

        if self.transform:
            transformed = self.transform(image=image, bboxes=labels)
            image = transformed['image']
            labels = transformed['bboxes']

        return image, labels

def analyze_dataset(data_path: str, classes):
    data_path = Path(data_path)
    labels_path = data_path / 'labels'
    class_distribution = Counter()

    for label_file in labels_path.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                class_distribution[class_id] += 1

    print("\nClass Distribution")
    for class_id, count in class_distribution.items():
        print(f"Class {classes[class_id]}: {count}")

    return class_distribution
