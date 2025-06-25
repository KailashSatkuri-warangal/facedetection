from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    """Train a YOLOv8 model on the dataset with emotion-only classes."""
    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')  # Nano model for faster training; use yolov8m.pt for better accuracy
    
    # Train the model
    logging.info("Starting training...")
    model.train(
        data="../dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="facial_expression_model",
        project="../models/runs",
        device=0  # Use GPU if available, else set to -1 for CPU
    )
    logging.info("Training completed.")

if __name__ == "__main__":
    train_model()