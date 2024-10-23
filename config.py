from dataclasses import dataclass
from typing import List
import yaml

@dataclass
class ModelConfig:
    img_size: int = 416
    batch_size: int = 16
    epochs: int = 20
    model_type: str = 'yolov8m.pt'
    lr0: float = 0.01
    weight_decay: float = 0.0005
    patience: int = 20
    classes: List[str] = ('glove', 'goggles', 'helmet', 'mask', 'no-suit',
                         'no_glove', 'no_goggles', 'no_helmet', 'no_mask',
                         'no_shoes', 'shoes', 'suit')

# Notice this function is NOT indented under the class
def load_config(yaml_path='/content/drive/MyDrive/ppe_detection/config/data.yaml'):
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    return ModelConfig(), data_config
