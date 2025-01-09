import os


class Config:
    def __init__(
        self,
        dataset_root_dir: str = "datasets/SceneTrailTrain",
        sava_processed_data_dir: str = "datasets/yolo_data",
        random_seed: int = 0,
        val_size=0.2,
        test_size=0.125,
        is_shuffle=True,
    ):
        self.dataset_root_dir = os.path.normpath(dataset_root_dir)
        self.save_processed_data_dir = os.path.normpath(sava_processed_data_dir)
        self.random_seed = random_seed
        self.val_size = val_size
        self.test_size = (test_size,)
        self.is_shuffle = is_shuffle
