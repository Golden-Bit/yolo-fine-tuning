import json

# Load the COCO JSON file
with open("../example_1/coco_annotations.json", "r") as file:
    coco_data = json.load(file)

# YOLO output directory
output_dir = "../example_1/output_labels_bbox/"

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

            # Calculate YOLO format: center_x, center_y, width, height
            x_min, y_min, box_width, box_height = bbox
            center_x = x_min + box_width / 2
            center_y = y_min + box_height / 2

            # Normalize values by image dimensions
            center_x /= image_width
            center_y /= image_height
            box_width /= image_width
            box_height /= image_height

            # Append the YOLO formatted string (class_id, center_x, center_y, width, height)
            yolo_annotations.append(f"{category_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n")

    # Save the YOLO annotations to a .txt file
    with open(f"{output_dir}/{image_filename.replace('.png', '.txt')}", "w") as yolo_file:
        yolo_file.writelines(yolo_annotations)
