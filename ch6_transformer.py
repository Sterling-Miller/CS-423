import warnings
import numpy as np
import pandas as pd
from typing_extensions import Self
from sklearn.impute import KNNImputer
from typing import Optional, Iterable, Literal
from sklearn.base import BaseEstimator, TransformerMixin


### Chapter 6 Custom Transformer:
class CustomKNNTransformer(BaseEstimator, TransformerMixin):
  """Imputes missing values using KNN.

  This transformer wraps the KNNImputer from scikit-learn and hard-codes
  add_indicator to be False. It also ensures that the input and output
  are pandas DataFrames.

  Parameters
  ----------
  n_neighbors : int, default=5
      Number of neighboring samples to use for imputation.
  weights : {'uniform', 'distance'}, default='uniform'
      Weight function used in prediction. Possible values:
      "uniform" : uniform weights. All points in each neighborhood
      are weighted equally.
      "distance" : weight points by the inverse of their distance.
      in this case, closer neighbors of a query point will have a
      greater influence than neighbors which are further away.
  """

  def __init__(self, n_neighbors: int = 5, weights: Literal['uniform', 'distance'] = 'uniform') -> None:
    self.n_neighbors = n_neighbors
    self.weights = weights
    self.columns = []
    self.imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, add_indicator=False)

  def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
    assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.fit expected DataFrame but got {type(X)}."
    if self.imputer.n_neighbors >= len(X):
        warnings.warn(f"{self.__class__.__name__}: `n_neighbors` is {self.imputer.n_neighbors} which is >= number of rows in dataset ({len(X)}). \n")
    
    self.columns = X.columns.to_list()
    self.imputer.fit(X)
    return self

  def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    assert self.imputer is not None, f'This CustomKNNTransformer instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'
    assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.transform expected DataFrame but got {type(X)}."
    if list(X.columns) != self.columns:
      warnings.warn(f"{self.__class__.__name__}: Column mismatch. KNNImputer requires the same columns during transform as during fit. \n")

    X_ = X.copy()
    for col in self.columns:
      if col not in X_:
        X_[col] = np.nan
    X_ = X_[self.columns]

    imputed_array = self.imputer.transform(X_)
    return pd.DataFrame(imputed_array, columns=self.columns)

  def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
    self.fit(X, y)
    result: pd.DataFrame = self.transform(X)
    return result
