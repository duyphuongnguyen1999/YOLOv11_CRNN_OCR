import os
from typing import Optional


class Config:
    def __init__(
        self,
        dataset_root_dir: str = "datasets/SceneTrialTrain",
        sava_processed_data_dir: str = "datasets/yolo_data",
        random_seed: Optional[int] = 0,
        val_size: Optional[float] = 0.2,
        test_size: Optional[float] = 0.125,
        is_shuffle: Optional[bool] = True,
    ):
        self.dataset_root_dir = os.path.normpath(dataset_root_dir)
        self.save_processed_data_dir = os.path.normpath(sava_processed_data_dir)
        self.random_seed = random_seed
        self.val_size = val_size
        self.test_size = test_size
        self.is_shuffle = is_shuffle
