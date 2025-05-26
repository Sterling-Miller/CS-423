import numpy as np
import pandas as pd
from typing_extensions import Self
from typing import Optional, Iterable
from sklearn.base import BaseEstimator, TransformerMixin


### Chapter 3 Custom Transformer:
class CustomPearsonTransformer(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer that removes highly correlated features
    based on Pearson correlation.

    Parameters
    ----------
    threshold : float
        The correlation threshold above which features are considered too highly correlated
        and will be removed.

    Attributes
    ----------
    correlated_columns : Optional[List[Hashable]]
        A list of column names (which can be strings, integers, or other hashable types)
        that are identified as highly correlated and will be removed.
    """

    def __init__(self, threshold: float) -> None:
        """
        Initialize the transformer with a correlation threshold.

        Parameters
        ----------
        threshold : float
            The correlation threshold for identifying highly correlated features.
        """
        assert isinstance(threshold, float), f"{self.__class__.__name__} expected a float for threshold but got {type(threshold)}."
        self.threshold = threshold
        self.correlated_columns = None  # Will be computed in fit

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Identify highly correlated columns in the input DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to analyze for correlations.
        y : Optional[Iterable], default=None
            Ignored, exists for compatibility.

        Returns
        -------
        Self
            Returns self with correlated_columns attribute set.
        """
        assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.fit expected DataFrame but got {type(X)}."

        # Compute the correlation matrix
        corr_matrix = X.corr().abs()

        # Identify columns to drop based on the threshold
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.correlated_columns = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.threshold)]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove the highly correlated columns identified during fitting.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to transform.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with correlated columns removed.
        """
        assert self.correlated_columns is not None, f"{self.__class__.__name__}.transform called before fit."
        assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.transform expected DataFrame but got {type(X)}."

        # Drop the correlated columns
        return X.drop(columns=self.correlated_columns, errors='ignore')
