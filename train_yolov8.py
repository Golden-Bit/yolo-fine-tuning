# Install the YOLOv8 library
# pip install ultralytics==8.0.28

# Import the YOLO package from Ultralytics
from ultralytics import YOLO

# Define the paths to the bbox_dataset and configuration files
DATA_YAML_PATH = 'path/to/your/bbox_dataset.yaml'  # Replace with your actual bbox_dataset.yaml path
MODEL_PATH = 'yolov8s-seg.pt'  # Use YOLOv8's pre-trained segmentation model

# Initialize the YOLOv8 model for segmentation
model = YOLO(MODEL_PATH)

# Train the model for segmentation
model.train(
    data=DATA_YAML_PATH,   # Path to bbox_dataset YAML file
    epochs=50,             # Number of epochs to train
    imgsz=640,             # Image size (default is 640x640)
    batch=16,              # Batch size
    name='yolov8_segmentation_experiment',  # Name of the experiment folder
    task='segment'         # Specify that it's a segmentation task
)

# To train for object detection (bounding boxes), simply change the model path
MODEL_PATH_DETECT = 'yolov8s.pt'  # Use YOLOv8's pre-trained detection model

# Initialize YOLOv8 model for bounding box detection
model_detect = YOLO(MODEL_PATH_DETECT)

# Train the model for object detection (bounding boxes)
model_detect.train(
    data=DATA_YAML_PATH,   # Path to bbox_dataset YAML file
    epochs=50,             # Number of epochs to train
    imgsz=640,             # Image size (default is 640x640)
    batch=16,              # Batch size
    name='yolov8_bounding_box_experiment',  # Name of the experiment folder
    task='detect'          # Specify that it's a detection task
)

# Validate the trained segmentation model
model.val(
    data=DATA_YAML_PATH,   # Path to bbox_dataset YAML file
    task='segment'         # Validate on segmentation
)

# Validate the trained detection model
model_detect.val(
    data=DATA_YAML_PATH,   # Path to bbox_dataset YAML file
    task='detect'          # Validate on object detection
)

# Predict on new images using the trained segmentation model
model.predict(
    source='path/to/your/test/images',  # Path to test images
    save=True,                          # Save the output predictions
    task='segment'                      # Predict on segmentation
)

# Predict on new images using the trained detection model
model_detect.predict(
    source='path/to/your/test/images',  # Path to test images
    save=True,                          # Save the output predictions
    task='detect'                       # Predict on object detection
)
