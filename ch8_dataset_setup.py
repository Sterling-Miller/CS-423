from sklearn.model_selection import train_test_split

### Chapter 8 dataset setup functions:
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
