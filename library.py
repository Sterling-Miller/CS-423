from __future__ import annotations  # must be first line in your library!
from typing import Dict, Any, Optional, Union, List, Set, Hashable, Literal, Tuple, Self, Iterable
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import warnings
import sklearn
import types

# Pass pandas tables through pipeline instead of numpy matrices
sklearn.set_config(transform_output="pandas") 

### Chapter 1 Custom Transformers:
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
        assert isinstance(threshold, float), f"{self.__class__.__name__} expected a float for threshold but got {type(threshold)}."
        self.threshold = threshold
        self.correlated_columns = None  # Will be computed in fit

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.fit expected DataFrame but got {type(X)}."

        # Compute the correlation matrix
        corr_matrix = X.corr().abs()

        # Identify columns to drop based on the threshold
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.correlated_columns = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.threshold)]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.correlated_columns is not None, f"{self.__class__.__name__}.transform called before fit."
        assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.transform expected DataFrame but got {type(X)}."

        # Drop the correlated columns
        return X.drop(columns=self.correlated_columns, errors='ignore')


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


### Chapter 7 Custom Transformer:
class CustomTargetTransformer(BaseEstimator, TransformerMixin):
    """
    A target encoder that applies smoothing and returns np.nan for unseen categories.

    Parameters:
    -----------
    col: name of column to encode.
    smoothing : float, default=10.0
        Smoothing factor. Higher values give more weight to the global mean.
    """

    def __init__(self, col: str, smoothing: float =10.0):
        self.col = col
        self.smoothing = smoothing
        self.global_mean_ = None
        self.encoding_dict_ = None

    def fit(self, X, y):
        """
        Fit the target encoder using training data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Target values.
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.fit expected Dataframe but got {type(X)} instead.'
        assert self.col in X, f'{self.__class__.__name__}.fit column not in X: {self.col}. Actual columns: {X.columns}'
        assert isinstance(y, Iterable), f'{self.__class__.__name__}.fit expected Iterable but got {type(y)} instead.'
        assert len(X) == len(y), f'{self.__class__.__name__}.fit X and y must be same length but got {len(X)} and {len(y)} instead.'

        #Create new df with just col and target - enables use of pandas methods below
        X_ = X[[self.col]]
        target = self.col+'_target_'
        X_[target] = y

        # Calculate global mean
        self.global_mean_ = X_[target].mean()

        # Get counts and means
        counts = X_[self.col].value_counts().to_dict()    #dictionary of unique values in the column col and their counts
        means = X_[target].groupby(X_[self.col]).mean().to_dict() #dictionary of unique values in the column col and their means

        # Calculate smoothed means
        smoothed_means = {}
        for category in counts.keys():
            n = counts[category]
            category_mean = means[category]
            # Apply smoothing formula: (n * cat_mean + m * global_mean) / (n + m)
            smoothed_mean = (n * category_mean + self.smoothing * self.global_mean_) / (n + self.smoothing)
            smoothed_means[category] = smoothed_mean

        self.encoding_dict_ = smoothed_means

        return self

    def transform(self, X):
        """
        Transform the data using the fitted target encoder.
        Unseen categories will be encoded as np.nan.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.
        """

        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.encoding_dict_, f'{self.__class__.__name__}.transform not fitted'

        X_ = X.copy()

        # Map categories to smoothed means, naturally producing np.nan for unseen categories, i.e.,
        # when map tries to look up a value in the dictionary and doesn't find the key, it automatically returns np.nan. That is what we want.
        X_[self.col] = X_[self.col].map(self.encoding_dict_)

        return X_

    def fit_transform(self, X, y):
        """
        Fit the target encoder and transform the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Target values.
        """
        return self.fit(X, y).transform(X)

### Chapter 7 Helper Function:
def find_random_state(
    features_df: pd.DataFrame,
    labels: Iterable,
    transformer: TransformerMixin,
    n: int = 200
                  ) -> Tuple[int, List[float]]:
    """
    Finds an optimal random state for train-test splitting based on F1-score stability.

    This function iterates through `n` different random states when splitting the data,
    applies a transformation pipeline, and trains a K-Nearest Neighbors classifier.
    It calculates the ratio of test F1-score to train F1-score and selects the random
    state where this ratio is closest to the mean.

    Parameters
    ----------
    features_df : pd.DataFrame
        The feature dataset.
    labels : Union[pd.Series, List]
        The corresponding labels for classification (can be a pandas Series or a Python list).
    transformer : TransformerMixin
        A scikit-learn compatible transformer for preprocessing.
    n : int, default=200
        The number of random states to evaluate.

    Returns
    -------
    rs_value : int
        The optimal random state where the F1-score ratio is closest to the mean.
    Var : List[float]
        A list containing the F1-score ratios for each evaluated random state.

    Notes
    -----
    - If the train F1-score is below 0.1, that iteration is skipped.
    - A higher F1-score ratio (closer to 1) indicates better train-test consistency.
    """

    model = KNeighborsClassifier(n_neighbors=5)
    Var: List[float] = []  # Collect test_f1/train_f1 ratios

    for i in range(n):
        train_X, test_X, train_y, test_y = train_test_split(
            features_df, labels, test_size=0.2, shuffle=True,
            random_state=i, stratify=labels  # Works with both lists and pd.Series
        )

        # Apply transformation pipeline
        transform_train_X = transformer.fit_transform(train_X, train_y)
        transform_test_X = transformer.transform(test_X)

        # Train model and make predictions
        model.fit(transform_train_X, train_y)
        train_pred = model.predict(transform_train_X)
        test_pred = model.predict(transform_test_X)

        train_f1 = f1_score(train_y, train_pred)

        if train_f1 < 0.1:
            continue  # Skip if train_f1 is too low

        test_f1 = f1_score(test_y, test_pred)
        f1_ratio = test_f1 / train_f1  # Ratio of test to train F1-score

        Var.append(f1_ratio)

    mean_f1_ratio: float = np.mean(Var)
    rs_value: int = np.abs(np.array(Var) - mean_f1_ratio).argmin()  # Index of value closest to mean

    return rs_value, Var


### Pipelines for Titanic and Customer datasets:
titanic_transformer = Pipeline(steps=[
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('map_class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('target_joined', CustomTargetTransformer(col='Joined', smoothing=10)),
    ('tukey_age', CustomTukeyTransformer(target_column='Age', fence='outer')),
    ('tukey_fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ('scale_age', CustomRobustTransformer(target_column='Age')),
    ('scale_fare', CustomRobustTransformer(target_column='Fare')),
    ('impute', CustomKNNTransformer(n_neighbors=5)),
    ], verbose=True)

customer_transformer = Pipeline(steps=[
    ('map_os', CustomMappingTransformer('OS', {'Android': 0, 'iOS': 1})),
    ('target_isp', CustomTargetTransformer(col='ISP')),
    ('map_level', CustomMappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('tukey_age', CustomTukeyTransformer('Age', 'inner')),  #from chapter 4
    ('tukey_time spent', CustomTukeyTransformer('Time Spent', 'inner')),  #from chapter 4
    ('scale_age', CustomRobustTransformer(target_column='Age')), #from 5
    ('scale_time spent', CustomRobustTransformer(target_column='Time Spent')), #from 5
    ('impute', CustomKNNTransformer(n_neighbors=5)),
    ], verbose=True)

### Best Splits for Titanic and Customer datasets:
titanic_variance_based_split = 107
customer_variance_based_split = 113

def dataset_setup(original_table, label_column_name:str, the_transformer, rs, ts=.2):
  # Separate features (X) and label (y)
  X = original_table.drop(columns=[label_column_name])
  y = original_table[label_column_name]

  # Use stratify for classification splits, if y is appropriate
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=ts, random_state=rs, shuffle=True, stratify=y
  )

  # Fit on training data and transform both train & test
  X_train_transformed = the_transformer.fit_transform(X_train, y_train)
  X_test_transformed = the_transformer.transform(X_test)

  # Convert to NumPy arrays
  x_train_numpy = X_train_transformed.to_numpy()
  x_test_numpy = X_test_transformed.to_numpy()
  y_train_numpy = y_train.to_numpy()
  y_test_numpy = y_test.to_numpy()

  return x_train_numpy, x_test_numpy, y_train_numpy, y_test_numpy

def titanic_setup(titanic_table, transformer=titanic_transformer, rs=titanic_variance_based_split, ts=.2):
  return dataset_setup(titanic_table, 'Survived', transformer, rs, ts)

def customer_setup(customer_table, transformer=customer_transformer, rs=customer_variance_based_split, ts=.2):
  return dataset_setup(customer_table, 'Rating', transformer, rs, ts)

def threshold_results(thresh_list, actuals, predicted):
  result_df = pd.DataFrame(columns=['threshold', 'precision', 'recall', 'f1', 'accuracy', 'auc'])

  for t in thresh_list:
    yhat = [1 if v >=t else 0 for v in predicted]
    precision = precision_score(actuals, yhat, zero_division=0)
    recall = recall_score(actuals, yhat, zero_division=0)
    f1 = f1_score(actuals, yhat)
    accuracy = accuracy_score(actuals, yhat)
    auc = roc_auc_score(actuals, predicted)
    result_df.loc[len(result_df)] = {'threshold':t, 'precision':precision, 'recall':recall, 'f1':f1, 'accuracy':accuracy, 'auc':auc}

  result_df = result_df.round(2)
  return (result_df, result_df.style.highlight_max(color = 'pink', axis = 0).format(precision=2))

def halving_search(model, grid, x_train, y_train, factor=2, min_resources="exhaust", scoring='roc_auc'):
  halving_cv = HalvingGridSearchCV(
      model, grid,
      scoring=scoring,
      n_jobs=-1,
      factor=factor,
      cv=5, random_state=1234,
      refit=True,
      min_resources=min_resources
  )

  grid_result = halving_cv.fit(x_train, y_train)
  return grid_result

def sort_grid(grid):
  sorted_grid = grid.copy()

  #sort values - note that this will expand range for you
  for k,v in sorted_grid.items():
    sorted_grid[k] = sorted(sorted_grid[k], key=lambda x: (x is None, x))  #handles cases where None is an alternative value

  #sort keys
  sorted_grid = dict(sorted(sorted_grid.items()))

  return sorted_grid

