
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib



def load_data(filename):
  df = pd.read_csv(filename, low_memory = False)
  data = df.shape
  print(f'Data Shape : {data}')
  return df


def split_input_output(data, target_column):
  X = data.drop(target_column, axis = 1)
  y = data[target_column]
  print(f'Original Data Shape : {data.shape}')
  print(f'X Data Shape : {X.shape}')
  print(f'y Data Shape : {y.shape}')
  return X, y


def split_train_test(X, y, test_size, random_state):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state =random_state, stratify=y)
  print(f'X_train Shape : {X_train.shape}')
  print(f'X_test Shape : {X_test.shape}')
  print(f'y_train Shape : {y_train.shape}')
  print(f'y_test Shape : {y_test.shape}')
  return X_train, X_test, y_train, y_test


def serialize_data(data, path):
  joblib.dump(data, path)


def deserialize_data(path):
  data = joblib.load(path)
  return data