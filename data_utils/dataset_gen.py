import os
import cv2
import albumentations as A
import numpy as np
from glob import glob

# Define the augmentation transformations to apply on the images
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Horizontal flip with 50% probability
    A.VerticalFlip(p=0.2),  # Vertical flip with 20% probability
    A.RandomRotate90(p=0.5),  # Random 90-degree rotation
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),  # Shift, scale, and rotate
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)  # Random brightness/contrast changes
])

# Input and output directories (to be defined by the user)
input_img_dir = "../example_1/input_data/"
input_mask = "example_1/mask.png"
output_img_dir = "../example_1/output_data/"
output_label_dir = "../example_1/output_labels_bbox/"

# Ensure that output directories exist
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)


# Function to save bounding box annotations in YOLO format
def save_bounding_box_annotation(mask, filename, output_label_dir):
    """
    Create and save bounding box annotations for the object in the image.
    The bounding box is calculated based on the object’s contours in the mask.

    Args:
        mask (numpy array): The binary mask of the object.
        filename (str): The base filename for the image.
        output_label_dir (str): The directory where the annotations should be saved.
    """
    mask_path = os.path.join(output_label_dir, filename.replace('.png', '.txt'))
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    with open(mask_path, 'w') as f:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = (x + w / 2) / width
            center_y = (y + h / 2) / height
            bbox_w = w / width
            bbox_h = h / height
            f.write(f"0 {center_x:.6f} {center_y:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")


# Function to save segmentation annotations in YOLO polygon format
def save_segmentation_annotation(mask, filename, output_label_dir):
    """
    Create and save segmentation annotations in polygon format for the object in the image.
    Each point of the object contour is normalized and saved.

    Args:
        mask (numpy array): The binary mask of the object.
        filename (str): The base filename for the image.
        output_label_dir (str): The directory where the annotations should be saved.
    """
    mask_path = os.path.join(output_label_dir, filename.replace('.png', '.txt'))
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    with open(mask_path, 'w') as f:
        for contour in contours:
            for point in contour:
                x, y = point[0]
                normalized_x = x / width
                normalized_y = y / height
                f.write(f"0 {normalized_x:.6f} {normalized_y:.6f} ")
            f.write("\n")  # Separate each contour with a newline


# Function to save both the augmented image and the corresponding annotation
def save_augmented_image(image, mask, filename, output_img_dir, output_label_dir, label_type="bbox"):
    """
    Save the augmented image and its corresponding annotation (bounding box or segmentation).

    Args:
        image (numpy array): The image to be saved.
        mask (numpy array): The binary mask of the object in the image.
        filename (str): The base filename for the image.
        output_img_dir (str): The directory where the augmented image should be saved.
        output_label_dir (str): The directory where the annotation should be saved.
        label_type (str): The type of annotation ('bbox' for bounding box or 'segmentation' for mask).
    """
    image_path = os.path.join(output_img_dir, filename)
    cv2.imwrite(image_path, image)

    if label_type == "bbox":
        save_bounding_box_annotation(mask, filename, output_label_dir)
    elif label_type == "segmentation":
        save_segmentation_annotation(mask, filename, output_label_dir)


# Load mask (binary mask assumed to be the same for all images)
mask = cv2.imread(input_mask, cv2.IMREAD_GRAYSCALE)

# Process each image in the input directory
label_type = "bbox"  # Choose between "bbox" (bounding boxes) or "segmentation" (polygons)

for img_path in glob(os.path.join(input_img_dir, "*.png")):
    img = cv2.imread(img_path)
    filename = os.path.basename(img_path)

    # Save the original image and its annotation
    save_augmented_image(img, mask, filename, output_img_dir, output_label_dir, label_type=label_type)

    # Apply augmentations and save augmented images
    for i in range(5):  # Number of augmentations per image
        augmented = transform(image=img, mask=mask)
        aug_img = augmented['image']
        aug_mask = augmented['mask']
        aug_filename = filename.replace(".jpg", f"_aug_{i}.jpg")

        save_augmented_image(aug_img, aug_mask, aug_filename, output_img_dir, output_label_dir, label_type=label_type)
