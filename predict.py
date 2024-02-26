import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

def predict_yolo(model, image_path, output_folder):
    image = Image.open(image_path)
    image_np = np.asarray(image)

    results = model(image_np)  # Make predictions on the image

    # Define your output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Output the results
    output_img, _ = plot_bboxes2(os.path.basename(image_path), output_folder, image_np, results[0].boxes.data, score=True, save=True)

def main():
    # Set up your YOLO model
    model = YOLO("runs/detect/train7/weights/best.pt")  # Update with your actual model path

    # Set up the image path and output folder
    image_path = "datasets/test_image.jpg"  # Update with your actual image path
    output_folder = "output_predictions"  # Update with your desired output folder

    # Make predictions
    predict_yolo(model, image_path, output_folder)

if __name__ == "__main__":
    main()
