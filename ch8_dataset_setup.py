from sklearn.model_selection import train_test_split


### Chapter 8 dataset setup functions:
def dataset_setup(original_table, label_column_name: str, the_transformer, rs, ts=0.2):
    """
    Splits a DataFrame into train and test sets, applies a transformer pipeline, and returns NumPy arrays.

    This function automates the typical steps for preparing a dataset for machine learning:
    - Separates features and labels.
    - Splits the data into training and test sets using a fixed random state and stratification.
    - Fits the provided transformer on the training data and applies it to both train and test sets.
    - Converts the resulting DataFrames to NumPy arrays for compatibility with scikit-learn estimators.

    Parameters
    ----------
    original_table : pd.DataFrame
        The full dataset including features and the label column.
    label_column_name : str
        The name of the column to use as the label (target variable).
    the_transformer : TransformerMixin
        A scikit-learn compatible transformer or pipeline to fit/transform the features.
    rs : int
        The random state to use for reproducible splitting.
    ts : float, default=0.2
        The proportion of the dataset to include in the test split.

    Returns
    -------
    x_train_numpy : np.ndarray
        Transformed training features as a NumPy array.
    x_test_numpy : np.ndarray
        Transformed test features as a NumPy array.
    y_train_numpy : np.ndarray
        Training labels as a NumPy array.
    y_test_numpy : np.ndarray
        Test labels as a NumPy array.

    Examples
    --------
    >>> x_train, x_test, y_train, y_test = dataset_setup(df, 'Survived', my_transformer, 42)
    >>> x_train.shape
    (712, 6)
    >>> y_train[:5]
    array([0, 1, 1, 0, 0])
    """
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
