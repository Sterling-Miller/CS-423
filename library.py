from __future__ import annotations  # must be first line in your library!
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List, Set, Hashable, Literal, Tuple, Self, Iterable
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from sklearn.pipeline import Pipeline
import sklearn

sklearn.set_config(transform_output="pandas")  # Pass pandas tables through pipeline instead of numpy matrices


class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that maps values in a specified column according to a provided dictionary.
    """

    def __init__(self, mapping_column: Union[str, int], mapping_dict: Dict[Hashable, Any]) -> None:
        assert isinstance(mapping_dict, dict), f"{self.__class__.__name__} expected a dictionary but got {type(mapping_dict)}."
        self.mapping_dict = mapping_dict
        self.mapping_column = mapping_column

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.transform expected DataFrame but got {type(X)}."
        assert self.mapping_column in X.columns, f"{self.__class__.__name__}.transform unknown column '{self.mapping_column}'."

        column_set = set(X[self.mapping_column].unique())
        keys_not_found = set(self.mapping_dict.keys()) - column_set
        keys_absent = column_set - set(self.mapping_dict.keys())

        if keys_not_found:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values: {keys_not_found}\n")
        if keys_absent:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values: {keys_absent}\n")

        X_ = X.copy()
        X_[self.mapping_column] = X_[self.mapping_column].replace(self.mapping_dict)
        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        return self.transform(X)


class CustomOHETransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that performs one-hot encoding on a specified column.
    """

    def __init__(self, target_column: str) -> None:
        assert isinstance(target_column, str), f"{self.__class__.__name__} expected a string but got {type(target_column)}."
        self.target_column = target_column

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.transform expected DataFrame but got {type(X)}."
        assert self.target_column in X.columns, f"{self.__class__.__name__}.transform unknown column '{self.target_column}'."

        return pd.get_dummies(
            X,
            columns=[self.target_column],
            prefix=self.target_column,
            prefix_sep='_',
            dummy_na=False,
            drop_first=False,
            dtype=int
        )


class CustomDropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that either drops or keeps specified columns in a DataFrame.
    """

    def __init__(self, column_list: List[str], action: Literal['drop', 'keep'] = 'drop') -> None:
        assert action in ['keep', 'drop'], f"{self.__class__.__name__} action must be 'keep' or 'drop', got '{action}'."
        assert isinstance(column_list, list), f"{self.__class__.__name__} expected a list but got {type(column_list)}."
        self.column_list = column_list
        self.action = action

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.transform expected DataFrame but got {type(X)}."

        unknown_columns = set(self.column_list) - set(X.columns)
        if unknown_columns:
            if self.action == 'keep':
                raise AssertionError(f"{self.__class__.__name__}.transform unknown columns to keep: {unknown_columns}")
            else:
                print(f"Warning: {self.__class__.__name__}.transform unknown columns to drop: {unknown_columns}.")

        if self.action == 'drop':
            return X.drop(columns=self.column_list, errors='ignore')
        else:
            return X[self.column_list]
