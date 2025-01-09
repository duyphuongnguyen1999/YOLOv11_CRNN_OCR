import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from config import Config


class DataProcessor:
    """
    A class to handle data loading, transformation, and saving for object detection datasets.

    Attributes:
        dataset_root_dir (str): The root directory containing the XML and image files.
        save_processed_data_dir (str): The directory to save processed data.
        img_paths (list(str)): List of file paths to the images.
        img_sizes (list(tuple)): List of image resolutions as (width, height).
        img_labels (Nested list): Nested list where each sublist contains the labels of tagged rectangles in the corresponding image.
        bounding_boxes  (Nested list): Nested list where each sublist contains bounding box coordinates for the corresponding image.
        yolo_data (list(tuple)): A list where each tuple contains (image_path, yolo_labels)
    """

    def __init__(self, config: Config):
        self.dataset_root_dir = config.dataset_root_dir
        self.save_processed_data_dir = config.save_processed_data_dir

        # Automatically run data processing pipeline
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
        self.img_paths, self.img_sizes, self.img_labels, self.bounding_boxes = (
            self._extract_data_from_xml(self.dataset_root_dir)
        )
        print("Data extraction completed.")

        # Step 2: Convert to YOLO format
        self.yolo_data = self._convert_to_yolo_format(
            self.img_paths, self.img_sizes, self.bounding_boxes
        )
        print("YOLO conversion completed.")

        # Step 3: Save processed data
        self._save_data(
            self.yolo_data, self.dataset_root_dir, self.save_processed_data_dir
        )
        print(f"Data saved to: {self.save_processed_data_dir}")

    def _extract_data_from_xml(self, config: Config):
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
        xml_path = os.path.join(config.dataset_root_dir, "words.xml")

        # Check if the XML file exists in the specified directory
        if not os.path.exists(xml_path):
            raise FileNotFoundError(
                f"File 'words.xml' not found in {config.dataset_root_dir}"
            )

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
            img_paths.append(os.path.join(self.dataset_root_dir, img_name))
            img_size = (int(resolution.attrib["x"]), int(resolution.attrib["y"]))
            img_sizes.append(img_size)

            # Iterate through each <taggedRectangles> element
            for bbs_list in tagged_rectangles:
                # Iterate through each <taggedRectangle> in <taggedRectangle>
                for bb in bbs_list:
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
    def __get_bounding_box(bb):
        """
        Extracts the bounding box coordinates (x, y, width, height) from the
        <taggedRectangle>.

        Parameters:
            bb (Element): The tagged rectangle element.

        Returns:
            list: A list of bounding box coordinates [x, y, width, height].
        """
        return [
            float(bb.attrib.get("x", 0)),
            float(bb.attrib.get("y", 0)),
            float(bb.attrib.get("width", 0)),
            float(bb.attrib.get("height", 0)),
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

    def _save_data(self, data, config: Config):
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
        save_dir = config.save_processed_data_dir
        root_dir = config.dataset_root_dir

        # Create the target directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Create subdirectories for images and labels
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

        # Iterate over each image path and its corresponding YOLO labels
        for image_path, yolo_labels in data:
            # Copy the image file to the 'images' subdirectory
            shutil.copy(
                os.path.join(root_dir, image_path),  # Full path to the source image
                os.path.join(save_dir, "images"),  # Destination 'images' folder
            )

            # Extract the base name of the image file (without extension)
            image_name = os.path.basename(image_path)
            image_name = os.path.splitext(image_name)[0]  # Remove the file extension

            # Create a text file for the labels corresponding to the image
            label_file_path = f"{image_name}.txt"

            # Save YOLO labels to the text file in the 'labels' subdirectory
            with open(os.path.join(save_dir, "labels", label_file_path), "w") as f:
                for label in yolo_labels:
                    f.write(f"{label}\n")
