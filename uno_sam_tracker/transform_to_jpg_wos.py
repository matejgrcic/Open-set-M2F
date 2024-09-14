import os
from PIL import Image

def convert_png_to_jpg(png_path, jpg_path):
    """Convert a PNG image to JPG format."""
    with Image.open(png_path) as img:
        rgb_img = img.convert('RGB')
        rgb_img.save(jpg_path, 'JPEG')

def transform_images_in_directory(root_dir):
    """Transform all PNG images in a directory and its subdirectories to JPG format."""
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.png'):
                png_path = os.path.join(dirpath, filename)
                jpg_path = os.path.splitext(png_path)[0] + '.jpg'
                convert_png_to_jpg(png_path, jpg_path)
                print(f"Converted {png_path} to {jpg_path}")

root_directory = '/mnt/sdb1/datasets/wos/raw_data_tmp'
transform_images_in_directory(root_directory)

