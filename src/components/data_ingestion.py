import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging


# ================= CONFIG CLASS =================
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

    # dataset path (matches your folder name)
    source_data_path: str = os.path.join("nootbook", "stud.csv")


# ================= DATA INGESTION =================
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method")

        try:
            # -------- READ DATASET --------
            df = pd.read_csv(self.ingestion_config.source_data_path)
            logging.info("Dataset read successfully")

            # -------- CREATE ARTIFACTS FOLDER --------
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            # -------- SAVE RAW DATA --------
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")

            # -------- TRAIN-TEST SPLIT --------
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


# ================= MAIN =================
if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    print("Train file path:", train_path)
    print("Test file path:", test_path)

    # ---------- CALL DATA TRANSFORMATION ----------
    data_transformation = DataTransformation()

    train_arr, test_arr, preprocessor_path = (
        data_transformation.initiate_data_transformation(train_path, test_path)
    )

    print("Train array shape:", train_arr.shape)
    print("Test array shape:", test_arr.shape)
    print("Preprocessor saved at:", preprocessor_path)




