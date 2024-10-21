import json
import os

# Load the COCO JSON file
with open("../example_1/coco_annotations.json", "r") as file:
    coco_data = json.load(file)

# YOLO output directory
output_dir = "../example_1/output_labels_seg"

# Process each annotation
for image_info in coco_data['images']:
    image_id = image_info['id']
    image_width = image_info['width']
    image_height = image_info['height']
    image_filename = image_info['file_name']

    yolo_annotations = []

    # Find the annotations for this image
    for ann in coco_data['annotations']:
        if ann['image_id'] == image_id:
            category_id = ann['category_id']  # YOLO class id
            bbox = ann['bbox']  # Bounding box from COCO format (x_min, y_min, width, height)
            segmentation = ann['segmentation'][0]  # Polygon segmentation data

            # Calculate YOLO format bounding box: center_x, center_y, width, height
            x_min, y_min, box_width, box_height = bbox
            center_x = x_min + box_width / 2
            center_y = y_min + box_height / 2

            # Normalize bounding box by image dimensions
            center_x /= image_width
            center_y /= image_height
            box_width /= image_width
            box_height /= image_height

            # Prepare polygon segmentation in YOLO format
            normalized_polygon = []
            for i in range(0, len(segmentation), 2):
                x = segmentation[i] / image_width
                y = segmentation[i + 1] / image_height
                normalized_polygon.append(f"{x:.6f} {y:.6f}")

            # Create YOLO segmentation annotation string
            yolo_annotation = f"{category_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f} " + " ".join(normalized_polygon) + "\n"
            yolo_annotations.append(yolo_annotation)

    # Save the YOLO annotations to a .txt file
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{image_filename.replace('.png', '.txt')}", "w") as yolo_file:
        yolo_file.writelines(yolo_annotations)
