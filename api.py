import pickle
from typing import Optional, Union, List, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from pathlib import Path
from ultralytics import YOLO
import io
from PIL import Image
import numpy as np

app = FastAPI(
    title="Anomaly Detection API",
    description="API for training, testing, and inference using the PaDiM model for anomaly detection. This API allows users to train a model, perform inference, and test the model's performance on various datasets."
)

# Define the paths to the bbox_dataset and configuration files
BBOX_DATA_YAML_PATH = 'example_1/bbox_data.yaml'  # Replace with your actual bbox_dataset.yaml path
SEG_DATA_YAML_PATH = 'example_1/seg_data.yaml'  # Replace with your actual seg_dataset.yaml path
MODEL_PATH = 'yolov8s-seg.pt'  # Use YOLOv8's pre-trained segmentation model
MODEL_PATH_DETECT = 'yolov8s.pt'  # Use YOLOv8's pre-trained detection model

# Initialize YOLOv8 models for segmentation and detection
model_seg = YOLO(MODEL_PATH)
model_detect = YOLO(MODEL_PATH_DETECT)

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    task: str = "segment"  # Can be "segment" or "detect"
):
    # Load the image from the uploaded file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Convert image to RGB if it has 4 channels (RGBA)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    elif image.mode != "RGB":
        raise HTTPException(status_code=400, detail="Unsupported image format. Only RGB or RGBA images are accepted.")

    # Check if the selected task is valid
    if task not in ["segment", "detect"]:
        raise HTTPException(status_code=400, detail="Invalid task type. Choose 'segment' or 'detect'.")

    # Choose the model based on the task
    model = model_seg if task == "segment" else model_detect

    # Perform prediction
    results = model.predict(source=np.array(image), save=False, task=task)

    # Format the results for the response based on the task type
    predictions = []
    for result in results:
        if task == "segment":
            # For segmentation, provide mask coordinates
            for mask in result.masks:
                predictions.append({
                    "mask": [elem.tolist() for elem in mask.xy],  # Convert mask coordinates to a list format
                    "confidence": result.boxes.conf.tolist(),
                    "class": result.boxes.cls.tolist()
                })
        elif task == "detect":
            # For detection, provide bounding box coordinates
            for box in result.boxes:
                predictions.append({
                    "box": box.xyxy.tolist(),
                    "confidence": box.conf.tolist(),
                    "class": box.cls.tolist()
                })

    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
