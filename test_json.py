import json
import pandas

with open(r'E:\Backup_K20pro\Download\treesat_benchmark\coco-panoptic\export_coco-panoptic_r_drone-panoramic_drone-panoramic.v01.json') as f:
    data = json.load(f)

# Extracting general information
info = data['info']
categories = data['categories']
images = data['images']
annotations = data['annotations']

print("Dataset description:", info['description'])
print("Dataset version:", info['version'])
print("\nCategories:")
for category in categories:
    print(f"ID: {category['id']}, Name: {category['name']}, Color: {category['color']}")

print("\nImages:")
for image in images:
    print(f"ID: {image['id']}, File Name: {image['file_name']}, Width: {image['width']}, Height: {image['height']}")

print("\nAnnotations:")
for annotation in annotations:
    print(f"Image ID: {annotation['image_id']}, File Name: {annotation['file_name']}")
    for segment_info in annotation['segments_info']:
        print(f"  Segment ID: {segment_info['id']}, Category ID: {segment_info['category_id']}, BBox: {segment_info['bbox']}, Area: {segment_info['area']}")

def get_images_by_category(category_name, data):
    category_id = None
    for category in data['categories']:
        if category['name'] == category_name:
            category_id = category['id']
            break
    
    if category_id is None:
        print(f"Category '{category_name}' not found.")
        return []

    images_with_category = []
    for annotation in data['annotations']:
        for segment_info in annotation['segments_info']:
            if segment_info['category_id'] == category_id:
                images_with_category.append(annotation['image_id'])
                break
    
    return images_with_category

# Example usage
category_name = "Attalea Maripa"
print("ATTALIA MARIPAAAAAAA")
image_ids = get_images_by_category(category_name, data)
print(image_ids)

