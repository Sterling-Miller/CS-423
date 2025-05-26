from sklearn.model_selection import HalvingGridSearchCV


def halving_search(model, grid, x_train, y_train, factor=3, min_resources="exhaust", scoring='roc_auc'):
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
