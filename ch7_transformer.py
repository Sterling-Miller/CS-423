import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from typing import Iterable, Tuple, List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


### Chapter 7 Custom Transformer:
class CustomTargetTransformer(BaseEstimator, TransformerMixin):
    """
    A target encoder that applies smoothing and returns np.nan for unseen categories.

    This transformer encodes a categorical column by replacing each category with a smoothed
    mean of the target variable, using the provided smoothing parameter. Unseen categories
    during transform are encoded as np.nan.

    Parameters
    ----------
    col : str
        Name of the column to encode.
    smoothing : float, default=10.0
        Smoothing factor. Higher values give more weight to the global mean.

    Attributes
    ----------
    col : str
        The column to encode.
    smoothing : float
        The smoothing factor used in encoding.
    global_mean_ : float
        The global mean of the target variable, computed during fitting.
    encoding_dict_ : dict
        Dictionary mapping categories to their smoothed means.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'cat': ['A', 'B', 'A', 'C'], 'target': [1, 0, 1, 0]})
    >>> enc = CustomTargetTransformer('cat', smoothing=5)
    >>> enc.fit(df, df['target'])
    >>> enc.transform(df)
       cat  target
    0  1.0       1
    1  0.2       0
    2  1.0       1
    3  0.2       0
    """

    def __init__(self, col: str, smoothing: float = 10.0):
        """
        Initialize the CustomTargetTransformer.

        Parameters
        ----------
        col : str
            Name of the column to encode.
        smoothing : float, default=10.0
            Smoothing factor for target encoding.
        """
        self.col = col
        self.smoothing = smoothing
        self.global_mean_ = None
        self.encoding_dict_ = None

    def fit(self, X, y):
        """
        Fit the target encoder using training data.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            Training data features.
        y : Iterable of shape (n_samples,)
            Target values.

        Returns
        -------
        Self
            The fitted transformer.
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.fit expected Dataframe but got {type(X)} instead.'
        assert self.col in X, f'{self.__class__.__name__}.fit column not in X: {self.col}. Actual columns: {X.columns}'
        assert isinstance(y, Iterable), f'{self.__class__.__name__}.fit expected Iterable but got {type(y)} instead.'
        assert len(X) == len(y), f'{self.__class__.__name__}.fit X and y must be same length but got {len(X)} and {len(y)} instead.'

        # Create new df with just col and target - enables use of pandas methods below
        X_ = X[[self.col]]
        target = self.col + '_target_'
        X_[target] = y

        # Calculate global mean
        self.global_mean_ = X_[target].mean()

        # Get counts and means
        counts = X_[self.col].value_counts().to_dict()
        means = X_[target].groupby(X_[self.col]).mean().to_dict()

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

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        pd.DataFrame
            DataFrame with the encoded column.
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.encoding_dict_, f'{self.__class__.__name__}.transform not fitted'

        X_ = X.copy()
        X_[self.col] = X_[self.col].map(self.encoding_dict_)
        return X_

    def fit_transform(self, X, y):
        """
        Fit the target encoder and transform the input data.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            Training data features.
        y : Iterable of shape (n_samples,)
            Target values.

        Returns
        -------
        pd.DataFrame
            DataFrame with the encoded column.
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
    labels : Iterable
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
            random_state=i, stratify=labels
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
