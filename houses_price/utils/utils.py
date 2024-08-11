import pandas as pd
import numpy as np

from typing import Dict


def check_dup_null(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:

    for name, dataframe in dataframes.items():

        duplicate: numpy.int64 = dataframe.duplicated().sum()
        null_values: pd.Series = dataframe.isnull().sum().sort_values(ascending=False)
        percent = null_values / dataframe.isnull().count().sort_values(ascending=False)
        dup_missing_data: pd.DataFrame = pd.concat(
            [null_values, percent * 100], keys=["Total", "Null_percent"], axis=1
        )

        print(f"The {name} has {duplicate} duplicates")
        print(f"The {name} has {null_values.sum()} null values")
        print("=" * 40)

        dataframes[name] = dup_missing_data

    return dataframes
