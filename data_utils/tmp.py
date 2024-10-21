import os
import cv2
from glob import glob
import math


# Function to rotate and save images without cropping the output
def rotate_and_save_images(input_dir, output_dir, angle=90):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all image file paths from the directory
    image_paths = glob(os.path.join(input_dir, "*.jpg")) + glob(os.path.join(input_dir, "*.png"))

    for img_path in image_paths:
        # Read the image
        img = cv2.imread(img_path)

        # Get the original dimensions of the image
        (h, w) = img.shape[:2]

        # Calculate the new size of the image after rotation
        # This ensures the image won't get cut off after rotation
        radians = math.radians(angle)
        new_w = int(abs(w * math.cos(radians)) + abs(h * math.sin(radians)))
        new_h = int(abs(w * math.sin(radians)) + abs(h * math.cos(radians)))

        # Create the rotation matrix
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Adjust the translation part of the rotation matrix to keep the image centered
        matrix[0, 2] += (new_w - w) / 2
        matrix[1, 2] += (new_h - h) / 2

        # Perform the rotation with the new dimensions
        rotated_img = cv2.warpAffine(img, matrix, (new_w, new_h))

        # Save the rotated image to the output directory
        filename = os.path.basename(img_path)
        output_img_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_img_path, rotated_img)
        print(f"Rotated and saved: {output_img_path}")


# Example usage
input_img_dir = "../example_1/input_data"  # Path to the input directory
output_img_dir = "../example_1/output_data"  # Path to the output directory
rotate_and_save_images(input_img_dir, output_img_dir, angle=90)
