import xgboost
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

xgb_reg = XGBRegressor(n_estimators=40, max_depth=10, learning_rate=0.4)

xgb_reg.fit(X_train, y_train)
