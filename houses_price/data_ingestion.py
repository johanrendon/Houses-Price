import os
import sys

from typing import List, Dict
from dataclasses import dataclass

import pandas as pd


@dataclass
class DataIngestionConfig:
    origin_train_data_path: str = os.path.join(".", "data", "raw", "train.csv")
    origin_test_data_path: str = os.path.join(".", "data", "raw", "test.csv")
    destination_train_data_path: str = os.path.join(".", "data", "interim", "train.csv")
    destination_test_data_path: str = os.path.join(".", "data", "interim", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def load_data(self) -> Dict[str, pd.DataFrame]:
        train_df: pd.DataFrame = pd.read_csv(self.ingestion_config.origin_train_data_path)
        test_df: pd.DataFrame = pd.read_csv(self.ingestion_config.origin_test_data_path)
        dataframes: Dict[str, pd.DataFrame] = {
            "Train": train_df,
            "Test": test_df,
        }
        return dataframes

        # /home/johan/data_science_personal/houses_price/data/raw
