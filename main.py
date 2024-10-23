import sys
sys.path.append('/content/drive/MyDrive/ppe_detection')

from src.config import load_config
from src.dataset import analyze_dataset
from src.utils import verify_dataset
from src.train import train_model, validate_model
from src.inference import inference_pipeline, process_results
import wandb
import cv2

def main():
    # Load configurations
    config, data_config = load_config()

    # Verify and analyze dataset
    verify_dataset(data_config['train'])
    class_distribution = analyze_dataset(data_config['train'], config.classes)

    # Initialize wandb
    wandb.init(project="ppe-detection")

    try:
        # Train model
        model, training_results = train_model(config, data_config)

        # Save model
        model.save('/content/drive/MyDrive/ppe_detection/best_ppe_model.pt')

        # Example inference
        test_image_path = data_config['test'] + '/images/005303_jpg.rf.d88d9335996cf880d42a0754273536db.jpg'
        original_image = cv2.imread(test_image_path)
        if original_image is None:
            raise ValueError(f"Could not load image from {test_image_path}")

        test_results = inference_pipeline(
            model_path='best_ppe_model.pt',
            image_path=test_image_path
        )

        # Process and visualize results
        processed_image = process_results(test_results[0], original_image, model.names)

        # Save processed image
        cv2.imwrite('/content/drive/MyDrive/ppe_detection/test_results.jpg', processed_image)

        from google.colab.patches import cv2_imshow
        cv2_imshow(processed_image)

    finally:
        wandb.finish()

if __name__ == "__main__":
    main()
