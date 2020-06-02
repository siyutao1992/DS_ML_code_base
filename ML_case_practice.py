import numpy as np
from sklearn.datasets import load_boston
import pandas as pd

bos = load_boston()
df = pd.DataFrame(data= np.c_[bos['data'], bos['target']],
                     columns= list(map(str, bos['feature_names'])) + ['target'])

df['CHAS'] = df['CHAS'].astype('category')
df['RAD'] = df['RAD'].astype('category')

# print(df['CHAS'].unique())

df_quant = df.copy() # try converting CHAS to dummy vars
# df_quant.info()
df_quant = df_quant.drop('RAD', axis=1)
# df_quant.info()

df_quant = pd.get_dummies(df_quant)
# df_quant.info()

# do a regression task

from sklearn.model_selection import train_test_split
X = df_quant.drop('target', axis=1).values # extract X data
y = df_quant['target'].values # extract Y data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

# 1. try a linear model
# from sklearn import linear_model as lm
# reg = lm.LinearRegression()
# reg.fit(X_train, y_train)
# y_pred_tr = reg.predict(X_train)
# print(reg.score(X_train, y_pred_tr)) # R^2 of the model
# print(reg.score(X_test, y_test))

# 2. try a ridge regression
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import Ridge
# params = {'alpha':[0.000001, 0.00001, 0.0001]}
# ridge = Ridge(normalize=True)
# grid = GridSearchCV(estimator=ridge, param_grid=params, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
# grid.fit(X_train, y_train)
# best_hyperparams = grid.best_params_
# best_CV_score = grid.best_score_
# best_model = grid.best_estimator_
# print(best_model.score(X_test, y_test))
# print(best_CV_score); print(best_hyperparams)

# 3. try a lasso regression
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import Lasso
# params = {'alpha':[0.0001, 0.001, 0.01]}
# lasso = Lasso(normalize=True)
# grid = GridSearchCV(estimator=lasso, param_grid=params, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
# grid.fit(X_train, y_train)
# best_hyperparams = grid.best_params_
# best_CV_score = grid.best_score_
# best_model = grid.best_estimator_
# print(best_model.score(X_test, y_test))
# print(best_CV_score); print(best_hyperparams)

# 4. try a random forest model
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error as MSE
# params = {'n_estimators':[18, 20, 22]}
# rf = RandomForestRegressor(n_estimators=10, random_state=21)
# grid = GridSearchCV(estimator=rf, param_grid=params, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
# grid.fit(X_train, y_train)
# best_hyperparams = grid.best_params_
# best_CV_score = grid.best_score_
# best_model = grid.best_estimator_
# print(best_model.score(X_test, y_test))

# importances_rf = pd.Series(best_model.feature_importances_, index = df_quant.drop('target', axis=1).columns)
# sorted_importances_rf = importances_rf.sort_values()
# print(sorted_importances_rf)

#############################################
df_qual = df.copy() # use CHAS as target var for classification
df_qual = df_qual.drop('RAD', axis=1)

X = df_qual.drop('CHAS', axis=1).values # extract X data
y = pd.to_numeric(df_qual['CHAS'].values) # extract Y data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# 1. try logistic regression
# from sklearn.linear_model import LogisticRegression
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)
# y_pred_prob = logreg.predict_proba(X_test)[:,1]
# print(logreg.score(X_test, y_test))
# # use AUC to assess the model
# from sklearn.metrics import roc_auc_score
# print(roc_auc_score(y_test, y_pred_prob)) # or use cv as shown below

# 2. try adaptive boosting tree
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.metrics import roc_auc_score
# dt = DecisionTreeClassifier(random_state=21)
# adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=50)
# adb_clf.fit(X_train, y_train)
# y_pred_proba = adb_clf.predict_proba(X_test)[:,1]
# print(adb_clf.score(X_test, y_test))
# print(roc_auc_score(y_test, y_pred_proba))

# 3. try random forest model
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
params = {'n_estimators':[22, 25, 30, 35]}
rf = RandomForestClassifier(n_estimators=10, random_state=21)
grid = GridSearchCV(estimator=rf, param_grid=params, scoring='accuracy', cv=10, n_jobs=-1)
grid.fit(X_train, y_train)
best_hyperparams = grid.best_params_
best_CV_score = grid.best_score_
best_model = grid.best_estimator_
print(best_model.score(X_test, y_test))

importances_rf = pd.Series(best_model.feature_importances_, index = df_qual.drop('CHAS', axis=1).columns)
sorted_importances_rf = importances_rf.sort_values()
print(sorted_importances_rf)