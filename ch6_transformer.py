import warnings
import numpy as np
import pandas as pd
from typing_extensions import Self
from sklearn.impute import KNNImputer
from typing import Optional, Iterable, Literal
from sklearn.base import BaseEstimator, TransformerMixin


### Chapter 6 Custom Transformer:
class CustomKNNTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that imputes missing values using k-nearest neighbors (KNN).

    This transformer wraps the scikit-learn KNNImputer and ensures that both input and output
    are pandas DataFrames. It is useful for filling in missing values in a dataset by averaging
    the values of the nearest neighbors.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighboring samples to use for imputation.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction.
        - "uniform": All points in each neighborhood are weighted equally.
        - "distance": Closer neighbors of a query point have a greater influence than those further away.

    Attributes
    ----------
    n_neighbors : int
        Number of neighbors used for imputation.
    weights : str
        Weight function used in prediction.
    columns : list
        List of column names seen during fitting.
    imputer : KNNImputer
        The underlying scikit-learn KNNImputer instance.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
    >>> knn = CustomKNNTransformer(n_neighbors=2)
    >>> imputed_df = knn.fit_transform(df)
    >>> imputed_df
         A    B
    0  1.0  4.0
    1  2.0  5.0
    2  3.0  4.5
    """

    def __init__(self, n_neighbors: int = 5, weights: Literal['uniform', 'distance'] = 'uniform') -> None:
        """
        Initialize the CustomKNNTransformer.

        Parameters
        ----------
        n_neighbors : int, default=5
            Number of neighboring samples to use for imputation.
        weights : {'uniform', 'distance'}, default='uniform'
            Weight function used in prediction.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.columns = []
        self.imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, add_indicator=False)

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit the KNN imputer on the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit the imputer.
        y : Optional[Iterable], default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        Self
            The fitted transformer.
        """
        assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.fit expected DataFrame but got {type(X)}."
        if self.imputer.n_neighbors >= len(X):
            warnings.warn(f"{self.__class__.__name__}: `n_neighbors` is {self.imputer.n_neighbors} which is >= number of rows in dataset ({len(X)}). \n")
        self.columns = X.columns.to_list()
        self.imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the input DataFrame using the fitted KNN imputer.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to transform.

        Returns
        -------
        pd.DataFrame
            A DataFrame with missing values imputed.
        """
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
        """
        Fit to data, then transform it in one step. Combines fit() and transform().

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit and transform.
        y : Optional[Iterable], default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pd.DataFrame
            A DataFrame with missing values imputed.
        """
        self.fit(X, y)
        result: pd.DataFrame = self.transform(X)
        return result
