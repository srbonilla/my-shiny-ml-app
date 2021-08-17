import numpy as np
from sklearn.datasets import make_regression # Generate fetures, outputs, and true coefficient
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load # to save and load model

# load training data
def load_train_data(X_fn="data/X.npy", y_fn="data/y.npy"):
  X = np.load(X_fn)
  y = np.load(y_fn)
  return X.reshape(-1,), y
  
# make prediction given a model
def make_prediction(X, model_fn='data/model.joblib'):
  
  # check for right dimensions
  assert X.ndim == 2 # matrix like
  assert X.shape[1] == 1 # 1 column
    
  # load model
  model = load(model_fn)
    
  # make prediction
  y_predict = model.predict(X)
    
  return y_predict
  
# generate prediciton curve
def generate_prediction_curve(xmin=None, xmax=None, X_fn="data/X.npy", y_fn="data/y.npy"):
  # set default xmin or xmax
  if xmin is None:
    xmin = np.min(np.load(X_fn))
    
  if xmax is None:
    xmax = np.max(np.load(X_fn))
  
  # generate data
  X = np.linspace(xmin, xmax, 200)
  
  # put in right dimensions
  X = X.reshape(-1, 1)
  
  # make prediction
  y = make_prediction(X)
  
  return X.reshape(-1,), y
  
# process input from shiny app into input to model
def process_x_string(x_string):
    
  x_list = x_string.split(", ")
  x_array = np.array(x_list, dtype='float')
  return x_array.reshape(-1,1)
  
# process output from model to string
def process_y_array(A1_y):
  
  # round to 2 decimals
  A1_y = np.round(A1_y, 2)
  
  # convert to string
  y_string = ', '.join(A1_y.astype(str))
  
  return y_string
  


