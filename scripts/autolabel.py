import os
from inference_sdk import InferenceHTTPClient
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="DAf4GJbDIebTEkqgmw7a"
)

# Define paths and emotion-only classes
DATASET_ROOT = "../dataset"
SPLITS = ['train', 'valid', 'test']
EMOTION_CLASSES = ['Boredom', 'Confusion', 'Engaged', 'Frustration', 'Sleepy', 'Yawning']  # Excluding 'face' and 'fake face'

def convert_to_yolo_format(prediction, img_width, img_height):
    """Convert Roboflow API prediction to YOLOv8 format, filtering for emotions only."""
    yolo_labels = []
    for pred in prediction.get('predictions', []):
        class_name = pred.get('class')
        if class_name not in EMOTION_CLASSES:
            logging.warning(f"Class {class_name} ignored (not an emotion class).")
            continue
        
        class_id = EMOTION_CLASSES.index(class_name)
        x_center = pred['x'] / img_width
        y_center = pred['y'] / img_height
        width = pred['width'] / img_width
        height = pred['height'] / img_height
        
        # YOLOv8 format: <class_id> <x_center> <y_center> <width> <height>
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_labels

def process_images():
    """Process all images in the dataset and generate YOLOv8 labels for emotions only."""
    for split in SPLITS:
        image_dir = os.path.join(DATASET_ROOT, split, 'images')
        label_dir = os.path.join(DATASET_ROOT, split, 'labels')
        
        # Create labels directory if it doesn't exist
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
            logging.info(f"Created directory: {label_dir}")
        
        # Process each image
        for image_name in os.listdir(image_dir):
            if not image_name.lower().endswith(('.jpg', '.jpeg', 'png')):
                logging.warning(f"Skipping non-image file: {image_name}")
                continue
            
            image_path = os.path.join(image_dir, image_name)
            logging.info(f"Processing {image_path}...")
            
            # Infer using Roboflow API
            try:
                result = CLIENT.infer(image_path, model_id="face-detector-k3mlx/1")
            except Exception as e:
                logging.error(f"Error processing {image_path}: {e}")
                continue
            
            # Get image dimensions
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                logging.error(f"Error opening image {image_path}: {e}")
                continue
            
            # Convert predictions to YOLO format
            yolo_labels = convert_to_yolo_format(result, img_width, img_height)
            
            # Save labels to file
            label_file = os.path.join(label_dir, image_name.rsplit('.', 1)[0] + '.txt')
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_labels))
            logging.info(f"Saved labels to {label_file}")

if __name__ == "__main__":
    process_images()