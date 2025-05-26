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

# Import custom transformers and helper from separate modules:
from custom_transformers import (
    CustomMappingTransformer,
    CustomOHETransformer,
    CustomDropColumnsTransformer,
    CustomPearsonTransformer,
    CustomSigma3Transformer,
    CustomTukeyTransformer,
    CustomRobustTransformer,
    CustomKNNTransformer,
    CustomTargetTransformer,
    find_random_state
)

# Pass pandas tables through pipeline instead of numpy matrices
sklearn.set_config(transform_output="pandas") 

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

  headers = {
  "selector": "th:not(.index_name)",
  "props": "background-color: #800000; color: white; text-align: center"
  }
  properties = {"border": "1px solid black", "width": "65px", "text-align": "center"}

  fancy_df = result_df.style.highlight_max(color = 'pink', axis = 0).format(precision=2).set_properties(**properties).set_table_styles([headers])
  return (result_df, fancy_df)

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
