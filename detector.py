import os
from ultralytics import YOLO
from config import DetectorConfig, DataProcessorConfig
from data_processor import DataProcessor


class Detector:
    def __init__(self, config: DetectorConfig, run_pipeline: bool = True):
        self.model_name = config.model_name
        self.model = YOLO(self.model_name)  # Load a model
        self.data = config.data
        self.epochs = config.epochs
        self.imgsz = config.imgsz
        self.cache = config.cache
        self.device = config.device
        self.patience = config.patience
        self.plots = config.plots

        if run_pipeline:
            self.run_pipeline()

    def run_pipeline(self):
        print("Start training pipeline...")
        # Train model
        self.train_results = self.model.train(
            data=self.data,  # Path to dataset YAML
            epochs=self.epochs,  # Number of training epochs
            imgsz=self.imgsz,  # Training image size
            cache=self.cache,
            device=self.device,
            patience=self.patience,
            plots=self.plots,
        )
        print("Train completed")

        # Evaluate the model's performance on the validation set
        curr_dir = os.getcwd()
        print("Load trained model...")
        model_path = os.path.join(curr_dir, "runs/detect/train/weights/best.pt")
        print("Load completed")
        self.model = YOLO(model_path)
        print("Start evaluation pipeline...")
        self.metrices = self.model.val()
        print("Evaluation complete")


if __name__ == "__main__":
    # Run DataProcessor pipeline
    data_processor_config = DataProcessorConfig()
    processor = DataProcessor(
        config=data_processor_config, run_ocr_data_processor=False
    )
    # Run training and evaluating pipeline
    detector_config = DetectorConfig()
    detector = Detector(detector_config)
