import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


# ================= CONFIG CLASS =================
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

    # 🔴 Updated to match YOUR real folder name
    source_data_path: str = os.path.join("nootbook", "stud.csv")


# ================= DATA INGESTION CLASS =================
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")

        try:
            # -------- READ DATASET --------
            df = pd.read_csv(self.ingestion_config.source_data_path)
            logging.info("Dataset read successfully as DataFrame")

            # -------- CREATE ARTIFACTS FOLDER --------
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            # -------- SAVE RAW DATA --------
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to artifacts folder")

            # -------- TRAIN TEST SPLIT --------
            logging.info("Train-test split initiated")

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # -------- SAVE SPLIT FILES --------
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    print("Train file path:", train_path)
    print("Test file path:", test_path)





