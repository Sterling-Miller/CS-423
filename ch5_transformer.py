import pandas as pd
from typing import Optional, Iterable, Self
from sklearn.base import BaseEstimator, TransformerMixin


### Chapter 5 Custom Transformer:
class CustomRobustTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies robust scaling to a specified column in a pandas DataFrame.

    This transformer calculates the interquartile range (IQR) and median
    during the `fit` method and then uses these values to scale the
    target column in the `transform` method. It is useful for reducing the influence
    of outliers when scaling features.

    Parameters
    ----------
    target_column : str
        The name of the column to be robustly scaled.

    Attributes
    ----------
    target_column : str
        The name of the column to be scaled.
    iqr : Optional[float]
        The interquartile range of the target column, computed during fitting.
    med : Optional[float]
        The median of the target column, computed during fitting.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'feature': [1, 2, 3, 100]})
    >>> scaler = CustomRobustTransformer('feature')
    >>> scaled_df = scaler.fit_transform(df)
    >>> scaled_df
       feature
    0    -0.5
    1     0.0
    2     0.5
    3    48.5
    """

    def __init__(self, target_column: str) -> None:
        """
        Initialize the CustomRobustTransformer.

        Parameters
        ----------
        target_column : str
            The name of the column to be robustly scaled.
        """
        assert isinstance(target_column, str), f"{self.__class__.__name__} expected a string but got {type(target_column)}."
        self.target_column = target_column
        self.iqr: Optional[float] = None
        self.med: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Compute the IQR and median for the specified target column.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data containing the target column.
        y : Optional[Iterable], default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        Self
            The fitted transformer.
        """
        assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.fit expected DataFrame but got {type(X)}."
        assert self.target_column in X.columns, f"{self.__class__.__name__}.fit unknown column '{self.target_column}'."
        X_ = X.copy()
        X_[self.target_column] = X_[self.target_column].astype(float)

        self.iqr = X_[self.target_column].quantile(0.75) - X_[self.target_column].quantile(0.25)
        self.med = X_[self.target_column].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply robust scaling to the target column using the computed IQR and median.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data containing the target column.

        Returns
        -------
        pd.DataFrame
            A copy of the input DataFrame with the target column robustly scaled.
        """
        assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.transform expected DataFrame but got {type(X)}."
        assert self.target_column in X.columns, f"{self.__class__.__name__}.transform unknown column '{self.target_column}'."
        assert self.iqr is not None and self.med is not None, f'This CustomRobustTransformer instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'

        X_ = X.copy()
        X_[self.target_column] = X_[self.target_column].astype(float)

        if self.iqr and self.iqr != 0:
            X_[self.target_column] = (X_[self.target_column] - self.med) / self.iqr
        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it in one step. Combines fit() and transform().

        Parameters
        ----------
        X : pandas.DataFrame
            The input data containing the target column.
        y : Optional[Iterable], default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pd.DataFrame
            A copy of the input DataFrame with the target column robustly scaled.
        """
        self.fit(X, y)
        result: pd.DataFrame = self.transform(X)
        return result
