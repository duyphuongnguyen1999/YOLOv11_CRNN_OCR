import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from config import SEED, VAL_SIZE, TEST_SIZE, IS_SHUFFLE


def extract_data_from_xml(root_dir):
    xml_path = os.path.join(root_dir, "words.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_paths = []
    img_sizes = []
    img_labels = []
    bounding_boxes = []

    for img in root:
        bbs_of_img = []
        labels_of_img = []

        for bbs in img.findall("taggedRectangles"):
            for bb in bbs:
                # Check non-alphabet and non-number
                if not bb[0].text.isalnum():
                    continue

                if "é" in bb[0].text.lower() or "ñ" in bb[0].text.lower():
                    continue

                bbs_of_img.append(
                    [
                        float(bb.attrib["x"]),
                        float(bb.attrib["y"]),
                        float(bb.attrib["width"]),
                        float(bb.attrib["height"]),
                    ]
                )
                labels_of_img.append(bb[0].text.lower())

        # Store
        img_path = os.path.join(root_dir, img[0].text)
        img_paths.append(img_path)
        img_size = (int(img[1].attrib["x"]), int(img[1].attrib["y"]))
        img_sizes.append(img_size)
        bounding_boxes.append(bbs_of_img)
        img_labels.append(labels_of_img)

    return img_paths, img_sizes, img_labels, bounding_boxes


def convert_to_yolo_format(image_paths, image_sizes, bounding_boxes):
    yolo_data = []

    for image_path, image_size, bboxes in zip(image_paths, image_sizes, bounding_boxes):
        image_width, image_height = image_size

        yolo_labels = []

        for bbox in bboxes:
            x, y, w, h = bbox

            # Calculate normalize bounding box coordinates
            center_x = (x + w / 2) / image_width
            center_y = (y + h / 2) / image_height
            normalize_width = w / image_width
            normalize_height = h / image_height

            # Because we only have 1 class, we set class_id to 0
            class_id = 0

            # Convert to YOLO format
            yolo_label = (
                f"{class_id} {center_x} {center_y} {normalize_width} {normalize_height}"
            )
            yolo_labels.append(yolo_label)

        yolo_data.append((image_path, yolo_labels))

    return yolo_data


def save_data(data, src_img_dir, save_dir):
    # Create folder if not exists
    os.makedirs(save_dir, exist_ok=True)

    # Make images and labels folder
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    for image_path, yolo_labels in data:
        # Copy image to images folder
        shutil.copy(
            os.path.join(src_img_dir, image_path), os.path.join(save_dir, "images")
        )

        # Save labels to lables folder
        image_name = os.path.basename(image_path)
        image_name = os.path.splitext(image_name)[0]

        label_file_path = f"{image_name}.txt"

        with open(os.path.join(save_dir, "labels", label_file_path), "w") as f:
            for label in yolo_labels:
                f.write(f"{label}\n")


def load_and_transform_data(dataset_dir, save_yolo_data_dir):
    img_paths, img_sizes, img_labels, bounding_boxes = extract_data_from_xml(
        dataset_dir
    )
    print(f"Number of images: {len(img_paths)}")
    print(f"Example image path: {img_paths[0]}")
    print(f"Example image size: {img_sizes[0]}")
    print(f"Example bounding boxes: {bounding_boxes[0][:2]}")
    print(f"Example labels: {img_labels[0][:2]}")

    # Define class labels
    class_labels = ["text"]

    # Convert data into YOLO format
    yolo_data = convert_to_yolo_format(
        image_paths=img_paths, image_sizes=img_sizes, bounding_boxes=bounding_boxes
    )

    # Train-Val-Test split
    train_data, test_data = train_test_split(
        yolo_data, test_size=VAL_SIZE, random_state=SEED, shuffle=IS_SHUFFLE
    )
    test_data, val_data = train_test_split(
        test_data, test_size=TEST_SIZE, random_state=SEED, shuffle=IS_SHUFFLE
    )

    # Save data
    os.makedirs(save_yolo_data_dir, exist_ok=True)
    save_train_dir = os.path.join(save_yolo_data_dir, "train")
    save_test_dir = os.path.join(save_yolo_data_dir, "test")
    save_val_dir = os.path.join(save_yolo_data_dir, "val")

    save_data(train_data, dataset_dir, save_train_dir)
    save_data(val_data, dataset_dir, save_val_dir)
    save_data(test_data, dataset_dir, save_test_dir)
