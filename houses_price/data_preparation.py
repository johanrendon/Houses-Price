import os
import sys

from typing import List, Dict, Set
from dataclasses import dataclass

from houses_price.data_ingestion import DataIngestion
from houses_price.utils.utils import check_dup_null

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer, MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


@dataclass
class DataPreparationConfig:
    train_data_path: str = os.path.join(".", "data", "interim", "train.csv")
    test_data_path: str = os.path.join(".", "data", "interim", "test.csv")


class DataPreparation:
    def __init__(self):
        self.preparation_config = DataPreparationConfig()

    def drop_processor(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:

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
            "GarageArea",
            "GarageCond",
            "BsmtFinType2",
            "BsmtExposure",
            "BsmtCond",
            "BsmtQual",
            "BsmtFinType1",
            "TotRmsAbvGrd",
            "1stFlrSF",
        ]

        for name, dataframe in dataframes.items():
            dataframe = dataframe.drop(columns_to_drop, axis=1)

            if name == "Train":
                dataframe.to_csv(self.preparation_config.train_data_path, index=False, header=True)
                dataframes[name] = dataframe

            else:
                dataframe.to_csv(self.preparation_config.test_data_path, index=False, header=True)
                dataframes[name] = dataframe

        return dataframes


@dataclass
class DataTransformConfig:
    processor_path: str = os.path.join(".", "data", "processor.pkl")
    train_data_path: str = os.path.join(".", "data", "interim", "transformed_train.csv")
    test_data_path: str = os.path.join(".", "data", "interim", "transformed_test.csv")


class DataTransform:
    def __init__(self, dataframes: Dict[str, pd.DataFrame]):
        self.transfor_config = DataTransformConfig()

        self._dataframes_list = list(dataframes.values())

        self.test_df: pd.DataFrame = self._dataframes_list[1].copy()
        self.train_df: pd.DataFrame = self._dataframes_list[0].copy()

        self.target: str = "SalePrice"
        self.cat_features = self.train_df.select_dtypes(include="object").columns
        self.num_features = self.train_df.select_dtypes(exclude="object")

        self.skewed_feats = (
            self.train_df[self.num_features.columns].drop([self.target, "Id"], axis=True).skew()
        )
        self.skewed_feats = self.skewed_feats[self.skewed_feats > 0.75].index

    def get_preprocessing_object(self):

        skew_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "log_transformer",
                    FunctionTransformer(
                        func=np.log1p, validate=True, feature_names_out="one-to-one"
                    ),
                ),
            ]
        )

        num_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(sparse_output=False)),
                ("scaler", MaxAbsScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                (
                    "num_pipeline",
                    num_pipeline,
                    self.num_features.drop([self.target, "Id"], axis=1).columns,
                ),
                ("cat_pipeline", cat_pipeline, self.cat_features),
            ],
            verbose_feature_names_out=False,
        )

        return preprocessor, skew_pipeline

    def init_data_transformation(self):

        preprocessing_obj, skew_pipeline = self.get_preprocessing_object()

        self.train_df[self.target] = np.log1p(self.train_df[self.target])

        self.train_df[self.skewed_feats] = skew_pipeline.fit_transform(
            self.train_df[self.skewed_feats]
        )
        self.test_df[self.skewed_feats] = skew_pipeline.transform(self.test_df[self.skewed_feats])

        preprocessing_obj.set_output(transform="pandas")
        transformed_train_df = preprocessing_obj.fit_transform(self.train_df)
        transformed_test_df = preprocessing_obj.transform(self.test_df)

        transformed_train_df = pd.concat(
            [transformed_train_df, self.train_df[self.target]], axis=1
        )

        transformed_test_df = pd.concat([transformed_test_df, self.test_df["Id"]], axis=1)

        transformed_train_df.to_csv(self.transfor_config.train_data_path, index=False, header=True)
        transformed_test_df.to_csv(self.transfor_config.test_data_path, index=False, header=True)

        return (transformed_train_df, transformed_test_df)


@dataclass
class FeatureSelectionConfig:
    train_feature_selection_path: str = os.path.join(".", "data", "processed", "train.csv")
    test_feature_selection_path: str = os.path.join(".", "data", "processed", "test.csv")


class FeatureSelection:
    def __init__(self):
        self.config = FeatureSelectionConfig()

    def get_features(self, dataframe: pd.DataFrame):
        self.X_train = dataframe.drop("SalePrice", axis=1).copy()
        self.y_train = dataframe["SalePrice"].copy()

        features_selection = SelectFromModel(estimator=Lasso(alpha=0.005, random_state=0))
        features_selection.fit(self.X_train, self.y_train)

        selected_features = self.X_train.columns[(features_selection.get_support())]

        return selected_features

    def select_features(self, dataframes: List[pd.DataFrame]):
        train_df = dataframes[0].copy()
        test_df = dataframes[1].copy()

        features = self.get_features(train_df)

        train_df = pd.concat([train_df[features], train_df['SalePrice']], axis=1)
        train_df.to_csv(self.config.train_feature_selection_path, index=False)

        test_df = pd.concat([test_df[features], test_df['Id']], axis=1)
        test_df.to_csv(self.config.test_feature_selection_path, index=False)


if __name__ == "__main__":

    data_ingestion = DataIngestion()
    dataframes: Dict[str, pd.DataFrame] = data_ingestion.load_data()

    data_preparation = DataPreparation()
    dataframes = data_preparation.drop_processor(dataframes)

    data_transformation = DataTransform(dataframes)
    df_train, df_test = data_transformation.init_data_transformation()

    feature_selection = FeatureSelection()
    feature_selection.select_features([df_train, df_test])
