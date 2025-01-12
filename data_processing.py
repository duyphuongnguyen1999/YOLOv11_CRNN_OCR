import os
import shutil
import yaml
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from config import DataProcessorConfig


class DataProcessor:
    """
    A class to handle data loading, transformation, and saving for object detection datasets.

    Attributes:
        dataset_root_dir (str): The root directory containing the XML and image files.
        save_yolo_data_dir (str): The directory to save processed data.
        img_paths (list(str)): List of file paths to the images.
        img_sizes (list(tuple)): List of image resolutions as (width, height).
        img_labels (Nested list): Nested list where each sublist contains the labels of tagged rectangles in the corresponding image.
        bounding_boxes  (Nested list): Nested list where each sublist contains bounding box coordinates for the corresponding image.
        yolo_data (list(tuple)): A list where each tuple contains (image_path, yolo_labels)
    """

    def __init__(self, config: DataProcessorConfig, run_pipeline: bool = True):
        self.config = config
        self.dataset_root_dir = config.dataset_root_dir
        self.save_yolo_data_dir = config.save_yolo_data_dir
        self.save_ocr_data_dir = config.save_ocr_data_dir
        self.random_seed = config.random_seed
        self.val_size = config.val_size
        self.test_size = config.test_size
        self.is_shuffle = config.is_shuffle
        self.class_labels = ["text"]

        # Automatically run data processing pipeline if specified
        if run_pipeline:
            self.run_pipeline()

    def run_pipeline(self):
        """
        Runs the full data processing pipeline:
        - Extracts data from XML.
        - Converts data to YOLO format.
        - Saves the processed data.
        """
        print("Starting data processing pipeline...")

        # Step 1: Extract data
        self.src_img_paths, self.img_sizes, self.img_labels, self.bounding_boxes = (
            self._extract_data_from_xml(self.dataset_root_dir)
        )
        print("Data extraction completed.")

        # Step 2: Convert to YOLO format
        self.yolo_data = self._convert_to_yolo_format(
            self.src_img_paths, self.img_sizes, self.bounding_boxes
        )
        print("YOLO conversion completed.")

        # Step 3: Split into train-test-val datasets and save processed data
        self.yolo_yaml_path, self.train_data, self.val_data, self.test_data = (
            self._train_val_test_split(self.yolo_data, self.config)
        )
        print(f"Data saved to: {self.save_yolo_data_dir}")

        # Step 4: Create ORC dataset
        self._split_bounding_boxes(
            self.src_img_paths,
            self.img_labels,
            self.bounding_boxes,
            self.save_ocr_data_dir,
        )

        # Step 5: Create OCR vocabulary
        self.ocr_vocab, self.ocr_vocab_size = self._build_vocabulary(
            self.ocr_labels, self.save_ocr_data_dir
        )

    def _extract_data_from_xml(self, dataset_root_dir):
        """
        Extracts data from a `words.xml` file located in the specified directory.

        The XML structure is expected to be as follows:
        ```
        <tagset>
            <image>
                <imageName>path/to/image</imageName>
                <resolution x="width" y="height" />
                <taggedRectangles>
                    <taggedRectangle x="x-coordinate" y="y-coordinate" width="width" height="height">
                        <tag>label</tag>
                        <segmentation />
                    </taggedRectangle>
                    ...
                </taggedRectangles>
            </image>
            ...
        </tagset>
        ```

        Parameters:
            dataset_root_dir (str): The root directory containing the XML and image files.

        Returns:
            tuple: A tuple containing the following:
                - img_paths (list of str): List of file paths to the images.
                - img_sizes (list of tuple): List of image resolutions as (width, height).
                - img_labels (list of list of str): Nested list where each sublist contains the labels
                of tagged rectangles in the corresponding image.
                - bboxes (list of list of list of float): Nested list where each sublist contains
                bounding box coordinates for the corresponding image.
                Each bounding box is represented as a list: [x, y, width, height].

        Notes:
            - Only bounding boxes with alphanumeric labels are included.
            - Any label containing specific characters (e.g., "é", "ñ") is ignored.
        """

        # Construct the full path to the words.xml file
        xml_path = os.path.join(dataset_root_dir, "words.xml")

        # Check if the XML file exists in the specified directory
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"File 'words.xml' not found in {xml_path}")

        # Parse the XML file and get the root element
        tree = ET.parse(xml_path)  # Load XML from file
        root = tree.getroot()  # Get the root element

        # Initialize empty lists to store data
        img_paths, img_sizes, img_labels, bounding_boxes = [], [], [], []

        # Iterate through each <image> element in the XML
        for image in root.findall("image"):
            # Initialize empty lists to store data
            bbs_of_img, labels_of_img = [], []

            # Extract image name, resolution, and tagged rectangles (bounding box)
            img_name = image.findtext("imageName")
            resolution = image.find("resolution")
            tagged_rectangles = image.find("taggedRectangles")

            # Skip missing data images (imageName, resolution, taggedRectangles)
            if img_name is None or resolution is None or tagged_rectangles is None:
                continue

            # Store the image file path and resolution (width, height)
            img_paths.append(img_name)
            img_size = (int(resolution.attrib["x"]), int(resolution.attrib["y"]))
            img_sizes.append(img_size)

            # Iterate through each <taggedRectangles> element
            for bb in tagged_rectangles:
                tag = bb.find("tag")
                if self.__is_valid_tag(tag):
                    # Extract the labels for each bounding box
                    labels_of_img.append(tag.text.lower())

                    # Extract the bounding box (x, y, width, height)
                    bbs_of_img.append(self.__get_bounding_box(bb))

            # Store
            bounding_boxes.append(bbs_of_img)
            img_labels.append(labels_of_img)

        return img_paths, img_sizes, img_labels, bounding_boxes

    @staticmethod
    def __is_valid_tag(tag):
        """
        Checks if the tag in the bounding box is valid
        (alphanumeric and does not contain invalid characters).

        Parameters:
            tag (Element): An XML element representing the tag.

        Returns:
            bool: True if the tag is valid, False otherwise.
        """

        return (
            # Ensure that each rectangle has a tag
            tag is not None
            and
            # Only include alphanumeric tags
            tag.text.isalnum()
            and
            # Exclude tags with specific characters
            "é" not in tag.text.lower()
            and "ñ" not in tag.text.lower()
        )

    @staticmethod
    def __get_bounding_box(tagged_rectangle):
        """
        Extracts the bounding box coordinates (x, y, width, height) from the
        <taggedRectangle>.

        Parameters:
            bb (Element): The tagged rectangle element.

        Returns:
            list: A list of bounding box coordinates [x, y, width, height].
        """
        return [
            float(tagged_rectangle.attrib.get("x", 0)),
            float(tagged_rectangle.attrib.get("y", 0)),
            float(tagged_rectangle.attrib.get("width", 0)),
            float(tagged_rectangle.attrib.get("height", 0)),
        ]

    def _convert_to_yolo_format(self, image_paths, image_sizes, bounding_boxes):
        """
        Convert bounding boxes data into YOLO format for object detection tasks.

        Parameters:
            image_paths (list of str): List of image file paths.
            image_sizes (list of tuple): List of image dimensions as (width, height).
            bounding_boxes (list of list of tuples): Nested list of bounding boxes for each image.
                Each bounding box is represented as (x, y, width, height).

        Returns:
            list of tuples: A list where each tuple contains:
                - image_path (str): The path to the image.
                - yolo_labels (list of str): List of YOLO formatted labels for each bounding box.
        """
        # Initialize an empty list to store YOLO formatted data
        yolo_data = []

        # Loop through each image and its corresponding size and bounding boxes
        for image_path, image_size, bboxes in zip(
            image_paths, image_sizes, bounding_boxes
        ):
            # Get the width and height of the image
            image_width, image_height = image_size

            # Initialize an empty list to store YOLO labels for this image
            yolo_labels = []

            # Loop through each bounding box in the current image
            for bbox in bboxes:
                x, y, w, h = bbox

                # Calculate normalize bounding box coordinates
                center_x = (x + w / 2) / image_width
                center_y = (y + h / 2) / image_height
                normalize_width = w / image_width
                normalize_height = h / image_height

                # Since we're assuming there's only 1 class (class_id = 0)
                class_id = 0  # Set the class ID for the object (0 for the first class)

                # Convert the bounding box information to YOLO format string
                yolo_label = (
                    f"{class_id} {center_x} {center_y} {normalize_width} "
                    f"{normalize_height}"
                )
                yolo_labels.append(yolo_label)  # Add the YOLO label to the list
            # Append the image path and its corresponding YOLO labels to the final output list
            yolo_data.append((image_path, yolo_labels))

        return yolo_data

    def _save_data(self, data, root_dir, save_dir):
        """
        Save YOLO format data to the specified directory.

        Parameters:
            data (list of tuple): A list where each element is a tuple containing:
                - image_path (str): Path to the image file relative to `src_img_dir`.
                - yolo_labels (list of str): List of YOLO format labels for the image.
            root_dir (str): Source directory containing the original image files.
            save_dir (str): Target directory to save processed images and labels.

        Returns:
            None
        """
        # Create the target directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Create subdirectories for images and labels
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

        # Iterate over each image path and its corresponding YOLO labels
        for image_path, yolo_labels in data:
            # Copy the image file to the 'images' subdirectory
            try:
                shutil.copy(
                    os.path.join(root_dir, image_path),  # Full path to the source image
                    os.path.join(save_dir, "images"),  # Destination 'images' folder
                )
            except FileNotFoundError as e:
                print(f"Error: File not found - {e}")
                continue

            # Extract the base name of the image file (without extension)
            image_name = os.path.basename(image_path)
            image_name = os.path.splitext(image_name)[0]  # Remove the file extension

            # Create a text file for the labels corresponding to the image
            label_file_path = os.path.join(save_dir, "labels", f"{image_name}.txt")

            # Save YOLO labels to the text file in the 'labels' subdirectory
            with open(label_file_path, "w") as f:
                for label in yolo_labels:
                    f.write(f"{label}\n")

    def _train_val_test_split(self, yolo_data, config: DataProcessorConfig):
        """
        Split data into train, val, and test sets, and save them into specified directory.
        Args:
            yolo_data (list): YOLO formatted data.
            config (Config): Configuration object.

        Returns:
            tuple: (train_data, val_data, test_data)
        """
        train_data, test_data = train_test_split(
            yolo_data,
            test_size=config.val_size,
            random_state=config.random_seed,
            shuffle=config.is_shuffle,
        )
        test_data, val_data = train_test_split(
            test_data,
            test_size=config.test_size,
            random_state=config.random_seed,
            shuffle=config.is_shuffle,
        )

        save_train_dir = os.path.join(config.save_yolo_data_dir, "train")
        save_val_dir = os.path.join(config.save_yolo_data_dir, "val")
        save_test_dir = os.path.join(config.save_yolo_data_dir, "test")

        self._save_data(train_data, config.dataset_root_dir, save_train_dir)
        self._save_data(val_data, config.dataset_root_dir, save_val_dir)
        self._save_data(test_data, config.dataset_root_dir, save_test_dir)

        data_yaml = {
            "path": "yolo_data/",
            "train": "train/images",
            "test": "test/images",
            "val": "val/images",
            "nc": 1,
            "name": self.class_labels,
        }
        yolo_yaml_path = os.path.join(config.save_yolo_data_dir, "data.yml")
        with open(yolo_yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        return yolo_yaml_path, train_data, val_data, test_data

    def _split_bounding_boxes(self, img_paths, img_labels, bboxes, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        count = 0
        labels = []  # List to store labels
        self.ocr_labels = []
        self.ocr_img_paths = []

        for img_path, img_label, bbs in zip(img_paths, img_labels, bboxes):
            img = Image.open(os.path.join(self.dataset_root_dir, img_path))

            for label, bb in zip(img_label, bbs):
                # Crop image
                cropped_img = img.crop((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))

                # Filter out if 90% of the cropped image is black or white
                if np.mean(cropped_img) < 35 or np.mean(cropped_img) > 220:
                    continue

                if cropped_img.size[0] < 10 or cropped_img.size[1] < 10:
                    continue

                # Save images
                filename = f"{count:06d}.jpg"
                new_img_path = os.path.join(save_dir, filename)
                cropped_img.save(new_img_path)
                self.ocr_img_paths.append(new_img_path)

                self.ocr_labels.append(label)
                label = new_img_path + "\t" + label

                labels.append(label)  # Append label to the list

                count += 1

        print(f"Created {count} images")

        # Write labels to a text file
        with open(os.path.join(save_dir, "labels.txt"), "w") as f:
            for label in labels:
                f.write(f"{label}\n")

    def _build_vocabulary(self, labels, save_voceb_dir):
        blank_char = "-"
        vocab = {blank_char: 0}

        index = 1

        unique_letters = set()

        # Iterate through all label in labels
        for label in labels:
            # Split label into letter
            label_letters = list(label.lower())
            unique_letters.update(label_letters)

        for letter in sorted(unique_letters):
            vocab[letter] = index
            index += 1

        vocab_size = len(vocab)

        # Write vocabulary into a text file
        with open(os.path.join(save_voceb_dir, "ocr_vocab.txt"), "w") as f:
            for letter, index in vocab.items():
                f.write(f"{index}\t{letter}\n")

        print(f"Build vocabulary successfully. Vocab size = {vocab_size}")

        return vocab, vocab_size
