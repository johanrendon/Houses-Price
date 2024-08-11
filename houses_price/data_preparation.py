import os
import sys

from typing import List, Dict, Set
from dataclasses import dataclass

from houses_price.data_ingestion import DataIngestion
from houses_price.utils.utils import check_dup_null

import pandas as pd


@dataclass
class DataPreparationConfig:
    train_data_path: str = os.path.join(".", "data", "interim", "train.csv")
    test_data_path: str = os.path.join(".", "data", "interim", "test.csv")


class DataPreparation:
    def __init__(self):
        self.preparation_config = DataPreparationConfig()

    def drop_processor(
        self, columns: List[str], dataframes: Dict[str, pd.DataFrame], features_fna: List[str]
    ) -> None:

        for name, dataframe in dataframes.items():
            dataframe = dataframe.drop(columns, axis=1)
            dataframe = dataframe.dropna(subset=features_fna)

            if name == "Train":
                dataframe.to_csv(self.preparation_config.train_data_path, index=False, header=True)

            else:
                dataframe.to_csv(self.preparation_config.test_data_path, index=False, header=True)


if __name__ == "__main__":

    columns_to_drop: List[str] = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "MasVnrType",
        "FireplaceQu",
        "LotFrontage",
        "GarageQual",
        "GarageFinish",
        "GarageType",
        "GarageYrBlt",
        "GarageCond",
        "BsmtFinType2",
        "BsmtExposure",
        "BsmtCond",
        "BsmtQual",
        "BsmtFinType1",
    ]

    features_fna: List[str] = ["MasVnrArea", "Electrical"]


    data_ingestion = DataIngestion()
    dataframes: Dict[str, pd.DataFrame] = data_ingestion.load_data()

    data_preparation = DataPreparation()
    data_preparation.drop_processor(columns_to_drop, dataframes, features_fna)
