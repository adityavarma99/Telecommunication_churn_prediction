import os
import sys
import dill

import numpy as np
import pandas as pd

import pickle

from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

## save object is from data transformation
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)


    
    