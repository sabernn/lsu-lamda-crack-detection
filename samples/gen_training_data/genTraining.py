"""
    Generates training dataset from paint-brushed dataset

    python3 genTraining.py --dataset=/path/to/dataset

    Expected Filesystem Structure:
    dataset <- Location as the optional --dataset argument
     ├── Annotations
     │ ├── 0.jpg
     │ ├── ...
     │ └── 9.jpg
     └── Images
       ├── 0.jpg
       ├── ...
       └── 9.jpg

    If you do not pass the --dataset argument, the program will use the filesystem
    structure located at \dataset
"""

import os
import cv2
import sys
import json
ROOT_DIR = os.path.abspath("../../")
print("Root directory is ", ROOT_DIR)

sys.path.append(ROOT_DIR)
from colorseg import extractcolor


def get_color(img):
    mask = extractcolor.get_nongrey_mask(img)
    target = cv2.bitwise_and(img, img, mask=mask)
    return target


def make_image_annotation(img, file_name, image_id, bbox):
    width, height = img.size
    image_annotation = {
        "file_name": file_name,
        "width": width,
        "height": height,
        "id": image_id
    }
    return image_annotation




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate training dataset from paint-brushed dataset'
    )

    parser.add_argument('--dataset', required=False,
                        default=ROOT_DIR + os.sep + 'dataset',
                        metavar="/path/to/dataset",
                        help='Directory where dataset is stored.')
    args = parser.parse_args()
    dataset_path = args.dataset
    annotations_path = dataset_path + os.sep + 'Annotations'
    images_path = dataset_path + os.sep + 'Images'

    print("Dataset Path: ", dataset_path)
    print("Annotations Path: ", annotations_path)
    print("Images Path: ", images_path)

    # Future color definition
    crack_id = [0]  # Later for future categories
    category_ids = {}  # Later for future categories

    is_crowd = 0
    annotation_id = 1
    image_id = 1

    annotations = []
    images = []
    categories = [
        {"supercategory": "material defect",
         "id": 1,
         "name": "crack"}
    ]

    for subdir, dirs, files in os.walk(annotations_path):
        for filename in files:
            filepath = subdir + os.sep + filename
            print("Loading file: %s" % filepath)
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)

            image = make_image_annotation(img, filename, image_id)
            images.append(image)

            bitmap = get_color(img)
            submasks = extractcolor.find_submasks(bitmap)
            for color, submask in submasks.items():
                # category_id = category_ids[image_id][color]
                category_id = 1  # Default for now
                annotation = extractcolor.make_submask_annotations(submask, image_id, category_id, annotation_id, is_crowd)
                annotations.append(annotation)
                annotation_id += 1

            image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    print(json.dumps(coco))