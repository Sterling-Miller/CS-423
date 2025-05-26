import numpy as np
import pandas as pd
from typing_extensions import Literal
from typing import Optional, Iterable, Self
from sklearn.base import BaseEstimator, TransformerMixin


### Chapter 4 Custom Transformers:
class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies 3-sigma clipping to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It clips values in the target column to be within three standard
    deviations from the mean.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply 3-sigma clipping on.

    Attributes
    ----------
    high_wall : Optional[float]
        The upper bound for clipping, computed as mean + 3 * standard deviation.
    low_wall : Optional[float]
        The lower bound for clipping, computed as mean - 3 * standard deviation.
    """

    def __init__(self, target_column: str) -> None:
        assert isinstance(target_column, str), f"{self.__class__.__name__} expected a string but got {type(target_column)}."
        self.target_column = target_column
        self.high_wall: Optional[float] = None
        self.low_wall: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.fit expected DataFrame but got {type(X)}."
        assert self.target_column in X.columns, f"{self.__class__.__name__}.fit unknown column '{self.target_column}'."
        assert X[self.target_column].dtype in [np.float64, np.float32], f"{self.__class__.__name__}.fit expected float column but got {X[self.target_column].dtype} instead."
        self.high_wall = X[self.target_column].mean() + 3 * X[self.target_column].std()
        self.low_wall = X[self.target_column].mean() - 3 * X[self.target_column].std()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.transform expected DataFrame but got {type(X)}."
        assert self.target_column in X.columns, f"{self.__class__.__name__}.transform unknown column '{self.target_column}'."
        assert self.high_wall is not None and self.low_wall is not None, f"{self.__class__.__name__}.transform high_wall and low_wall must be set by fit() first."

        X_ = X.copy()
        X_[self.target_column] = np.clip(X_[self.target_column], self.low_wall, self.high_wall)
        return X_
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        self.fit(X, y)  # Call the fit function
        result: pd.DataFrame = self.transform(X)
        return result
    
class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies Tukey's fences (inner or outer) to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in a scikit-learn pipeline.
    It clips values in the target column based on Tukey's inner or outer fences.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply Tukey's fences on.
    fence : Literal['inner', 'outer'], default='outer'
        Determines whether to use the inner fence (1.5 * IQR) or the outer fence (3.0 * IQR).

    Attributes
    ----------
    inner_low : Optional[float]
        The lower bound for clipping using the inner fence (Q1 - 1.5 * IQR).
    outer_low : Optional[float]
        The lower bound for clipping using the outer fence (Q1 - 3.0 * IQR).
    inner_high : Optional[float]
        The upper bound for clipping using the inner fence (Q3 + 1.5 * IQR).
    outer_high : Optional[float]
        The upper bound for clipping using the outer fence (Q3 + 3.0 * IQR).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'values': [10, 15, 14, 20, 100, 5, 7]})
    >>> tukey_transformer = CustomTukeyTransformer(target_column='values', fence='inner')
    >>> transformed_df = tukey_transformer.fit_transform(df)
    >>> transformed_df
    """

    def __init__(self, target_column: str, fence: Literal['inner', 'outer'] = 'outer') -> None:
        assert isinstance(target_column, str), f"{self.__class__.__name__} expected a string but got {type(target_column)}."
        assert fence in ['inner', 'outer'], f"{self.__class__.__name__} fence must be 'inner' or 'outer', got '{fence}'."
        self.target_column = target_column
        self.fence = fence
        self.inner_low: Optional[float] = None
        self.outer_low: Optional[float] = None
        self.inner_high: Optional[float] = None
        self.outer_high: Optional[float] = None
    
    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.fit expected DataFrame but got {type(X)}."
        assert self.target_column in X.columns, f"{self.__class__.__name__}.fit unknown column '{self.target_column}'."
        assert X[self.target_column].dtype in [np.float64, np.float32, np.int64], f"{self.__class__.__name__}.fit expected float column but got {X[self.target_column].dtype} instead."

        Q1 = X[self.target_column].quantile(0.25)
        Q3 = X[self.target_column].quantile(0.75)
        IQR = Q3 - Q1

        if self.fence == 'inner':
            self.inner_low = Q1 - 1.5 * IQR
            self.inner_high = Q3 + 1.5 * IQR
        else:
            self.outer_low = Q1 - 3.0 * IQR
            self.outer_high = Q3 + 3.0 * IQR
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.transform expected DataFrame but got {type(X)}."
        assert self.target_column in X.columns, f"{self.__class__.__name__}.transform unknown column '{self.target_column}'."
        assert (self.inner_low is not None and self.inner_high is not None) or (self.outer_low is not None and self.outer_high is not None), \
            f"{self.__class__.__name__}.transform inner/outer fences must be set by fit() first."

        X_ = X.copy()
        if self.fence == 'inner':
            X_[self.target_column] = np.clip(X_[self.target_column], self.inner_low, self.inner_high)
        else:
            X_[self.target_column] = np.clip(X_[self.target_column], self.outer_low, self.outer_high)
        return X_
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        self.fit(X, y)
        result: pd.DataFrame = self.transform(X)
        return result

