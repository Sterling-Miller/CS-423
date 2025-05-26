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
from ch2_transformers import (
  CustomDropColumnsTransformer,
  CustomMappingTransformer,
  CustomOHETransformer
)
from ch3_transformer import CustomPearsonTransformer
from ch4_transformers import (
  CustomSigma3Transformer,
  CustomTukeyTransformer
)
from ch5_transformer import CustomRobustTransformer
from ch6_transformer import CustomKNNTransformer
from ch7_transformer import (
    CustomTargetTransformer,
    find_random_state
)
from ch8_dataset_setup import dataset_setup
from ch10_threshold import threshold_results
from ch11_search import (
  halving_search,
  sort_grid
)

### Ch. 7 - Based Splits for Titanic and Customer datasets:
titanic_variance_based_split = 107
customer_variance_based_split = 113

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

### Ch. 9 - Dataset setup functions for Titanic and Customer datasets:
def titanic_setup(titanic_table, transformer=titanic_transformer, rs=titanic_variance_based_split, ts=.2):
  return dataset_setup(titanic_table, 'Survived', transformer, rs, ts)

def customer_setup(customer_table, transformer=customer_transformer, rs=customer_variance_based_split, ts=.2):
  return dataset_setup(customer_table, 'Rating', transformer, rs, ts)
