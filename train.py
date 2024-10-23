from ultralytics import YOLO
import torch
import wandb
import yaml
from pathlib import Path

def train_model(config, data_config):
    wandb.init(project='PPE-Detection', config=vars(config))

    # Create a temporary YAML file with the data configuration
    yaml_path = Path('temp_data_config.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f)

    model = YOLO(config.model_type)

    results = model.train(
        data=str(yaml_path),  # Pass the path to the YAML file as string
        epochs=config.epochs,
        imgsz=config.img_size,
        batch=config.batch_size,
        patience=config.patience,
        lr0=config.lr0,
        weight_decay=config.weight_decay,
        cache=True,
        device='0' if torch.cuda.is_available() else 'cpu',
        project='PPE-Detection',
        name='PPE_Detector',
        exist_ok=True,
        pretrained=True,
        optimizer='Adam'
    )

    # Clean up temporary file
    yaml_path.unlink(missing_ok=True)

    return model, results

def validate_model(model, valid_path):
    results = model.val(data=valid_path)
    return results
