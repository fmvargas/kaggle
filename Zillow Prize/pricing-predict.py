
# Math, stat and data
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# sklearn for regressions
from sklearn import ensemble, linear_model, clone
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
# keras for deep learning
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

# packages for geopatial regressions
# import pysal as ps
# import geopandas as gpd

# packages for viz
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image
import mplleaflet as mpll


import warnings
warnings.filterwarnings("ignore")
# In[]
try:
    path = 'C:\\OneDrive\\____Cursos\\Kaggle\\Zillow Prize\\data\\properties_2016.csv'
except:   
    pass

# In[]
df = pd.read_csv(path)
# In[]
df.head(10)
# In[]




# In[]



# In[]



# In[]



# In[]

# In[]