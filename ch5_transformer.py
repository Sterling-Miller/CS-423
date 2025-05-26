import pandas as pd
from typing import Optional, Iterable, Self
from sklearn.base import BaseEstimator, TransformerMixin


### Chapter 5 Custom Transformer:
class CustomRobustTransformer(BaseEstimator, TransformerMixin):
  """Applies robust scaling to a specified column in a pandas DataFrame.
    This transformer calculates the interquartile range (IQR) and median
    during the `fit` method and then uses these values to scale the
    target column in the `transform` method.

    Parameters
    ----------
    column : str
        The name of the column to be scaled.

    Attributes
    ----------
    target_column : str
        The name of the column to be scaled.
    iqr : float
        The interquartile range of the target column.
    med : float
        The median of the target column.
  """

  def __init__(self, target_column: str) -> None:
    assert isinstance(target_column, str), f"{self.__class__.__name__} expected a string but got {type(target_column)}."
    self.target_column = target_column
    self.iqr: Optional[float] = None
    self.med: Optional[float] = None

  def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
    assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.fit expected DataFrame but got {type(X)}."
    assert self.target_column in X.columns, f"{self.__class__.__name__}.fit unknown column '{self.target_column}'."
    X_ = X.copy()
    X_[self.target_column] = X_[self.target_column].astype(float)

    self.iqr = X_[self.target_column].quantile(0.75) - X_[self.target_column].quantile(0.25)
    self.med = X_[self.target_column].median()
    return self

  def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.transform expected DataFrame but got {type(X)}."
    assert self.target_column in X.columns, f"{self.__class__.__name__}.transform unknown column '{self.target_column}'."
    assert self.iqr is not None and self.med is not None, f'This CustomRobustTransformer instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'

    # Changed code: convert target column to float if needed
    X_ = X.copy()
    X_[self.target_column] = X_[self.target_column].astype(float)

    if self.iqr and self.iqr != 0:
        X_[self.target_column] = (X_[self.target_column] - self.med) / self.iqr
    return X_

  def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
    self.fit(X, y)
    result: pd.DataFrame = self.transform(X)
    return result
