from google.colab import drive
drive.mount('/content/gdrive')

import pandas as pd
import numpy as np

train = pd.read_csv(r"/content/gdrive/My Drive/geoanalytics/train.csv",dtype = np.float32)
test = pd.read_csv(r"/content/gdrive/My Drive/geoanalytics/test.csv",dtype = np.float32)

train_without_id = train.drop("id", axis = 1)

X_train = train_without_id.loc[:,train_without_id.columns != "score"].values
y_train = train_without_id.score.values

test_ids = test.id.values
test_ids = [int(id) for id in test_ids]
test_without_id = test.drop("id", axis = 1)
X_test = test_without_id.values

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

import xgboost
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

xgb_reg = XGBRegressor(n_estimators=40, max_depth=10, learning_rate=0.4)

xgb_reg.fit(X_train, y_train)
preds = xgb_reg.predict(X_val)

r2 = r2_score(y_val, preds)

y_pred = xgb_reg.predict(X_test)

df = pd.DataFrame({'id': test_ids, 'score': y_pred})

import csv
df.to_csv('sumission.csv', index = False)
