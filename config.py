import os
from typing import Optional, Union
import torch
from torchvision import transforms

ocr_data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((100, 420)),
            transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
            ),
            transforms.Grayscale(num_output_channels=1),
            transforms.GaussianBlur(3),
            transforms.RandomAffine(degrees=1, shear=1),
            transforms.RandomPerspective(
                distortion_scale=0.3,
                p=0.5,
                interpolation=3,
            ),
            transforms.RandomRotation(degrees=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((100, 420)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
}


class DataProcessorConfig:
    def __init__(
        self,
        dataset_root_dir: str = "datasets/SceneTrialTrain",
        sava_yolo_data_dir: str = "datasets/yolo_data",
        save_ocr_data_dir: str = "datasets/ocr_data",
        random_seed: int = 0,
        yolo_val_size: float = 0.2,
        yolo_test_size: float = 0.125,
        yolo_is_shuffle: bool = True,
        ocr_val_size: float = 0.1,
        ocr_test_size: float = 0.1,
        ocr_is_shuffle: bool = True,
        ocr_data_transforms=ocr_data_transforms,
        ocr_train_batch_size: int = 64,
        ocr_test_batch_size: int = 64 * 2,
    ):
        self.dataset_root_dir = os.path.normpath(dataset_root_dir)
        self.save_yolo_data_dir = os.path.normpath(sava_yolo_data_dir)
        self.save_ocr_data_dir = os.path.normpath(save_ocr_data_dir)
        self.random_seed = random_seed
        self.yolo_val_size = yolo_val_size
        self.yolo_test_size = yolo_test_size
        self.yolo_is_shuffle = yolo_is_shuffle
        self.ocr_val_size = ocr_val_size
        self.ocr_test_size = ocr_test_size
        self.ocr_is_shuffle = ocr_is_shuffle
        self.ocr_data_transforms = ocr_data_transforms
        self.ocr_train_batch_size = ocr_train_batch_size
        self.ocr_test_batch_size = ocr_test_batch_size


class DetectorConfig:
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        data: str = "datasets/yolo_data/data.yml",
        epochs: int = 100,
        imgsz: int = 640,
        cache: Union[str, bool] = "disk",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        patience: int = 20,
        plots: bool = True,
    ):
        self.model_name = model_name
        self.data = os.path.normpath(data)
        self.epochs = epochs
        self.imgsz = imgsz
        self.cache = cache
        self.device = device
        self.patience = patience
        self.plots = plots


class RecognitionConfig:
    def __init__(self, max_label_len: Optional[int] = 20):
        self.max_label_len = max_label_len
