from __future__ import annotations  # must be first line in your library!
import pandas as pd
import numpy as np
import types
from typing import Dict, Any, Optional, Union, List, Set, Hashable, Literal, Tuple, Self, Iterable
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from sklearn.pipeline import Pipeline
import sklearn

sklearn.set_config(transform_output="pandas")  # Pass pandas tables through pipeline instead of numpy matrices

class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that maps values in a specified column according to a provided dictionary.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It applies value substitution to a specified column using
    a mapping dictionary, which can be useful for encoding categorical variables or
    transforming numeric values.

    Parameters
    ----------
    mapping_column : str or int
        The name (str) or position (int) of the column to which the mapping will be applied.
    mapping_dict : dict
        A dictionary defining the mapping from existing values to new values.
        Keys should be values present in the mapping_column, and values should
        be their desired replacements.

    Attributes
    ----------
    mapping_dict : dict
        The dictionary used for mapping values.
    mapping_column : str or int
        The column (by name or position) that will be transformed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'category': ['A', 'B', 'C', 'A']})
    >>> mapper = CustomMappingTransformer('category', {'A': 1, 'B': 2, 'C': 3})
    >>> transformed_df = mapper.fit_transform(df)
    >>> transformed_df
       category
    0        1
    1        2
    2        3
    3        1
    """

    def __init__(self, mapping_column: Union[str, int], mapping_dict: Dict[Hashable, Any]) -> None:
        """
        Initialize the CustomMappingTransformer.

        Parameters
        ----------
        mapping_column : str or int
            The name (str) or position (int) of the column to apply the mapping to.
        mapping_dict : Dict[Hashable, Any]
            A dictionary defining the mapping from existing values to new values.

        Raises
        ------
        AssertionError
            If mapping_dict is not a dictionary.
        """
        assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
        self.mapping_dict: Dict[Hashable, Any] = mapping_dict
        self.mapping_column: Union[str, int] = mapping_column  #column to focus on

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomMappingTransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self  #always the return value of fit

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the mapping to the specified column in the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame or if mapping_column is not in X.

        Notes
        -----
        This method provides warnings if:
        1. Keys in mapping_dict are not found in the column values
        2. Values in the column don't have corresponding keys in mapping_dict
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
        warnings.filterwarnings('ignore', message='.*downcasting.*')  #squash warning in replace method below

        #now check to see if all keys are contained in column
        column_set: Set[Any] = set(X[self.mapping_column].unique())
        keys_not_found: Set[Any] = set(self.mapping_dict.keys()) - column_set
        if keys_not_found:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

        #now check to see if some keys are absent
        keys_absent: Set[Any] = column_set - set(self.mapping_dict.keys())
        if keys_absent:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

        X_: pd.DataFrame = X.copy()
        X_[self.mapping_column] = X_[self.mapping_column].replace(self.mapping_dict)
        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.
        """
        #self.fit(X,y)  #commented out to avoid warning message in fit
        result: pd.DataFrame = self.transform(X)
        return result


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
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        #self.fit(X,y)  #commented out to avoid warning message in fit
        result: pd.DataFrame = self.transform(X)
        return result


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
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        #self.fit(X,y)  #commented out to avoid warning message in fit
        result: pd.DataFrame = self.transform(X)
        return result

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
        assert X[self.target_column].dtype in [np.float64, np.float32], f"{self.__class__.__name__}.fit expected float column but got {X[self.target_column].dtype} instead."

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
    
titanic_transformer = Pipeline(steps=[
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('map_class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('ohe_joined', CustomOHETransformer(target_column='Joined')),
    ('tukey_age', CustomTukeyTransformer(target_column='Age', fence='outer')),
    ('tukey_fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ('scale_age', CustomRobustTransformer(target_column='Age')),
    ('scale_fare', CustomRobustTransformer(target_column='Fare')),
    ], verbose=True)

customer_transformer = Pipeline(steps=[
    ('drop', CustomDropColumnsTransformer(['ID'], 'drop')),
    ('gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('experience', CustomMappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high': 2})),
    ('os', CustomOHETransformer(target_column='OS')),
    ('isp', CustomOHETransformer(target_column='ISP')),
    ('time spent', CustomTukeyTransformer('Time Spent', 'inner')),
    ], verbose=True)