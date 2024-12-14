#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Configuration
CONFIG = {
    "model_path": "yolo11m.pt",  # Path to the YOLO model
    "image_path": "1EYFejGUjvjPcc4PZTwoufw.jpg",  # Path to the input image
    "confidence_threshold": 0.4,  # Confidence threshold for detection
}

# Load the YOLO model
def load_yolo_model(model_path):
    print(f"Loading YOLO model from: {model_path}")
    return YOLO(model_path)

# Perform YOLO inference
def run_yolo_inference(model, image_path, conf_threshold):
    print(f"Running YOLO inference on: {image_path}")
    return model(source=image_path, conf=conf_threshold)

# Visualize Results
def visualize_results(results):
    print("Visualizing results...")
    result_img = results[0].plot()[..., ::-1]  # Convert BGR to RGB
    plt.imshow(result_img)
    plt.axis("off")
    plt.show()

# Save Annotated Image
def save_annotated_image(results, save_path="annotated_image.jpg"):
    print(f"Saving annotated image to: {save_path}")
    annotated_img = results[0].plot()
    cv2.imwrite(save_path, annotated_img)

# Main Workflow
def main():
    model = load_yolo_model(CONFIG["model_path"])
    results = run_yolo_inference(model, CONFIG["image_path"], CONFIG["confidence_threshold"])
    visualize_results(results)
    save_annotated_image(results, save_path="annotated_image.jpg")

if __name__ == "__main__":
    main()

