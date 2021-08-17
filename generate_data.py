import numpy as np
from sklearn.datasets import make_regression # Generate fetures, outputs, and true coefficient
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load # to save and load model

def generate_data(n=200, X_fn="data/X.npy", y_fn="data/y.npy", model_fn="data/model.joblib"):
  
  # generate dummy data
  X, y, coef = make_regression(n_samples = n, 
                               n_features = 1, n_informative = 1,
                               n_targets = 1, noise = 10,
                               coef = True, random_state=420)

  # add non linear behaviour to the solution
  y[y > 0] = np.power(y[y > 0], 0.87)
  
  # save X and y
  np.save(X_fn, X)
  np.save(y_fn, y)
    
  # define and train simple ml model
  clf = RandomForestRegressor(max_depth=4)
  clf.fit(X, y)

  # save model
  dump(clf, model_fn)
  
# run program
if __name__ == "__main__":
  generate_data()
