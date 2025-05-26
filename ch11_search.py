from sklearn.model_selection import HalvingGridSearchCV


def halving_search(model, grid, x_train, y_train, factor=3, min_resources="exhaust", scoring='roc_auc'):
    """
    Performs a HalvingGridSearchCV to find the best hyperparameters for a model.

    This function uses scikit-learn's HalvingGridSearchCV to efficiently search over a grid of hyperparameters.
    It successively eliminates poor-performing candidates and allocates more resources to promising ones.

    Parameters
    ----------
    model : estimator object
        The machine learning estimator to tune.
    grid : dict
        Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
    x_train : array-like or pd.DataFrame
        Training features.
    y_train : array-like or pd.Series
        Training labels.
    factor : int, default=3
        The 'halving' parameter; how aggressively to cut down the candidate set.
    min_resources : int, str, or float, default="exhaust"
        Minimum resources to allocate to a candidate before elimination.
    scoring : str, default='roc_auc'
        Scoring metric to use for model evaluation.

    Returns
    -------
    grid_result : HalvingGridSearchCV object
        The fitted HalvingGridSearchCV object containing the results of the search.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> grid = {'n_estimators': [10, 50], 'max_depth': [3, 5]}
    >>> result = halving_search(RandomForestClassifier(), grid, x_train, y_train)
    >>> print(result.best_params_)
    """
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
    """
    Sorts a hyperparameter grid dictionary by keys and values for consistent ordering.

    This function sorts the values of each hyperparameter (handling None values) and then sorts the keys.

    Parameters
    ----------
    grid : dict
        Dictionary with parameter names as keys and lists of parameter values as values.

    Returns
    -------
    sorted_grid : dict
        A new dictionary with sorted keys and sorted values.

    Examples
    --------
    >>> grid = {'max_depth': [None, 3, 1], 'n_estimators': [50, 10]}
    >>> sort_grid(grid)
    {'max_depth': [1, 3, None], 'n_estimators': [10, 50]}
    """
    sorted_grid = grid.copy()

    # Sort values - handles cases where None is an alternative value
    for k, v in sorted_grid.items():
        sorted_grid[k] = sorted(sorted_grid[k], key=lambda x: (x is None, x))

    # Sort keys
    sorted_grid = dict(sorted(sorted_grid.items()))

    return sorted_grid
