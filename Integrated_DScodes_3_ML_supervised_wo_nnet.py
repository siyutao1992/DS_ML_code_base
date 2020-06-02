###############################################################################################
### Machine learning -- supervised (w/o nnet model)

import numpy as np
import pandas as pd

###############################################################################################
##################### bias & variance tradeoff ####################################################

# Q: When can we say fitted model suffer from high variance problem?
# A: when cv error of model much bigger than training error of model
#   Q: Remedy?
#   A: (1) decrease model complexity (2) get more data
# Q: When can we say fitted model suffer from high bias problem?
# A: When cv error of model is about the same as training error of model, but much bigger than desired error
#   Q: Remedy?
#   A: (1) increase model complexity (2) get more relevant features

###############################################################################################
############### classification goodness assessment ##################################

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred)) # output meaning is below
#                       predicted span          predict not spam
#   Actual spam:     [ True positive (tp),      False negative (fn)]
#   Actual not spam: [ False positive (fp),     True negative (tn)]
print(classification_report(y_test, y_pred)) # output sample is below
#       precision   recall  f1-score support
#   0   0.95        0.88    0.91        59
#   1   0.94        0.97    0.96        115
# avg/total 0.94    0.94    0.94        174
# NOTE: precision=tp/(tp+fp), recall=tp/(tp+fn), f1_score=2*(precision*recall)/(precision+recall)
# fpr = fp / (fp + tn), tpr = tp / (tp + fn)
accuracy_score(y_test, y_pred)

###############################################################################################
############### Utilities associated with modeling ##################################

#####################################
#### random sample/number generation
random_bs_samples = np.random.choice(arr, size=len(arr)) # draw (bootstrap) samples from arr with replacement
random_bs_inds = np.random.choice(np.arange(len(arr)), size=len(arr)) # draw (bootstrap) indices
    # can use these inds to draw bootstrap pairs of data, and compute other bootstrap stats (e.g., regression line)
permuted_data = np.random.permutation(arr) # can be used in A/B testing
    # for more details, refer to datacamp course "Statistical Thinking in Python (Part 2)"
# Hypothesis test of correlation
    # Posit null hypothesis: the two variables are completely uncorrelated
    # Simulate data assuming null hypothesis is true
    # Use Pearson correlation, ρ, as test statistic
    # Compute p-value as fraction of replicates that have ρ at least as large as observed.
random_samples = np.random.random(size=4) # generate random number from distribution U(0, 1)
bernoulli_samples = np.random.binomial(n=1, p=0.5, size=10) #
binomial_samples = np.random.binomial(n=4, p=0.5, size=10) 
poisson_samples = np.random.poisson(lam=2.0, size=100) 
    # number of events happened if they happen independently at a const rate
normal_samples = np.random.normal(mean=0.0, std=1.0, size=10000)
exp_sampels = np.random.exponential(scale=2.0, size=100) # waiting time between poisson events

#####################################
#### preprocessing data
df_numerical = pd.get_dummies(df) # convert categorical variables into dummy vars
# handle na by dropping
df['col_name'].replace(0, np.nan, inplace=True) # convert 0 values into nan values
df.dropna() # drop rows with nan values
# handle na with imputer
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)
# scale data
from sklearn.preprocessing import scale
X_scaled = scale(X)

#####################################
#### train/test split (for performance assessment of final model)
from sklearn.model_selection import train_test_split
X = df.drop('target_var_col', axis=1).values # extract X data
y = df['target_var_col'].values # extract Y data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

#####################################
#### cross-validation
from sklearn.model_selection import cross_val_score
cv_results = cross_val_score(model, X_train, y_train, cv=5) # e.g. model = regression model instance
    # By default cross_val_score uses the scoring provided in the given model, usually the simplest scoring method.
    # Most sklearn models provide a default scoring method.
print(cv_results.mean()) # mean value of the k cv errors in k-fold cv

#####################################
#### grid search cv 
# example of a classification tree model
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=SEED)
print(dt.get_params()) # print out all dt's hyperparameters as a dict
params_dt = { 'max_depth': [3, 4, 5, 6], 'min_samples_leaf': [0.04, 0.06, 0.08], 'max_features': [0.2, 0.4, 0.6, 0.8]}
grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, scoring='accuracy', cv=10, n_jobs=-1)
grid_dt.fit(X_train, y_train)
best_hyperparams = grid_dt.best_params_
best_CV_score = grid_dt.best_score_
best_model = grid_dt.best_estimator_
test_acc = best_model.score(X_test,y_test)

#####################################
#### pipelining data preprocessing and modeling (except train-test split)
from sklearn.pipeline import Pipeline
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
steps = [('imputation', imp), ('logistic_regression', LogisticRegression())]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
pipeline.score(X_test, y_test)
# scale with pipeline
from sklearn.preprocessing import StandardScaler
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
parameters = {knn__n_neighbors=np.arange(1, 50)}

#### CV + pipeline
cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
cv.best_params_
cv.score(X_test, y_test)
classification_report(y_test, y_pred)

###############################################################################################
############## Linear models (regression) #######################

#####################################
#### linear regression
# method 0: use numpy
slope, intercept = np.polyfit(x_vec, y_vec, 1)
# method 1: use statsmodels module
import statsmodels.api as sm
X_train = sm.add_constant(X_train) # NOTE WE NEED TO MANUALLY ADD CONSTANT TO X
model = sm.OLS(y_train, X_train).fit() # NOTE THE ORDER OF ARGUMENTS! 
    # The inputs can be either ndarray or dataframe
model.summary() # print out all summary statistics
y_pred = model.predict(X_test)
# method 2: use sklearn
from sklearn import linear_model as lm
reg = lm.LinearRegression()
reg.fit(X_train, y_train)
y_pred_tr = reg.predict(X_train)
reg.score(X_train, y_pred_tr) # R^2 of the model
coefs = reg.coef_ # a list of coefficients
intercept = reg.intercept_

#### polynomial regression
from sklearn import preprocessing as pp
reg = lm.LinearRegression()
poly = pp.PolynomialFeatures(2, include_bias=True)
reg.fit(poly.fit_transform(X_train), y_train)

#### ridge regression
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)

#### Lasso regression
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)
# feature selection
x_var_names = df.drop('target_var_col', axis=1).columns
lasso_coef = lasso.fit(X, y).coef_
_ = plt.plot(range(len(x_var_names)), lasso_coef)
_ = plt.xticks(range(len(x_var_names)), x_var_names, rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()

#####################################
#### logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)[:,1]
logreg.coef_ # print coefficients
# use ROC curve to assess goodness of logistic model
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate’)
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
# use AUC to assess goodness of logistic model
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_prob)) # or use cv as shown below
cv_results = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')

###############################################################################################
############## Nonlinear models (can do both regression and classification) #######################

#####################################
#### KNN model for classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn.score(X_test, y_test) # print the performance score of fitted model

#####################################
#### CART (classification and regression tree) - regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.1, random_state=3)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
mse_dt = MSE(y_test, y_pred)
rmse_dt = mse_dt**(1/2)
# using CV with regression tree needs some tweak below
#   (score is for maximization while MSE is for minimization)
MSE_CV = - cross_val_score(dt, X_train, y_train, cv= 10, scoring='neg_mean_squared_error', n_jobs = -1)
print(MSE_CV.mean())
#### CART (classification and regression tree) - classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dt = DecisionTreeClassifier(max_depth=2, min_samples_split=2, min_samples_leaf=1, \
                            criterion='gini', random_state=1) # decision tree
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred)

#####################################
#### Bagging regressor (individual models can be trees, nnet models, etc.)
# Quote from sklearn documentation: 
    # "bagging methods work best with strong and complex models (e.g., fully developed decision trees), ...
    # in contrast with boosting methods which usually work best with weak models (e.g., shallow decision trees)."
## NOTE: In theory, the mechanism of bagging can be used with any individual type of models, ...
    # but in practice it is almost always used with trees.
    # Therefore, while the module is "sklearn.ensemble", I regard it as a single type of model.
from sklearn.ensemble import BaggingRegressor
'details are omitted here. See classfication case below'
#### Bagging classifier (individual models can be trees, nnet models, etc.)
from sklearn.ensemble import BaggingClassifier
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=SEED)
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1)
bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
oob_accuracy = bc.oob_score_ # out-of-bag prediction accuracy score
# NOTE: with oob accuracy we may not need separate test data anymore

#####################################
#### random forest (tree) for regression
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=0.12, random_state=SEED)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rf.score(X_test, y_test) # this would be r^2 score for test data
mse_test = MSE(y_test, y_pred)
sse_test = MSE(y_test, np.mean(y_test)*np.ones(np.shape(y_test)))
print(1-mse_test/sse_test)
# can produce feature importance data to evaluate relative importance of different features
importances_rf = pd.Series(rf.feature_importances_, index = X.columns)
sorted_importances_rf = importances_rf.sort_values()
sorted_importances_rf.plot(kind='barh', color='lightgreen')
plt.show()
# Feature importance is calculated as the decrease in node impurity weighted by the 
    # probability of reaching that node. 
# The node probability can be calculated by the number of samples that reach the node,
    # divided by the total number of samples. The higher the value the more important the feature.
#### random forest (tree) for classification
from sklearn.ensemble import RandomForestClassifier
'details are omitted here. See regression case above'

#####################################
#### Adaboosting regressor
## NOTE: In theory, the mechanism of adaptive boosting can be used with any individual type of models, ...
    # but in practice it is almost always used with trees.
    # Therefore, while the module is "sklearn.ensemble", I regard it as a single type of model.
from sklearn.ensemble import AdaBoostRegressor
'details are omitted here. See classfication case below'
#### Adaboosting classifier
# this classification is done through voting, so can predict probability
from sklearn.ensemble import AdaBoostClassifier
dt = DecisionTreeClassifier(max_depth=1, random_state=SEED)
adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)
adb_clf.fit(X_train, y_train)
y_pred_proba = adb_clf.predict_proba(X_test)[:,1]
adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)

#####################################
#### Gradient boosting (tree) for regression
from sklearn.ensemble import GradientBoostingRegressor
gbt = GradientBoostingRegressor(n_estimators=300, max_depth=1, random_state=SEED)
gbt.fit(X_train, y_train)
y_pred = gbt.predict(X_test)
rmse_test = MSE(y_test, y_pred)**(1/2)
## Stochastic gradient boosting (tree) for regression
# just add "subsample" parameter to the gbt model. All else are the same.
sgbt = GradientBoostingRegressor(max_depth=1, subsample=0.8, \
                                max_features=0.2, n_estimators=300, random_state=SEED)
#### Gradient boosting (tree) for classification
from sklearn.ensemble import GradientBoostingClassifier
'details are omitted here. See regression case above'

###############################################################################################
############## Ensemble models (can do both regression and classification) #######################

#####################################
#### Voting regressor
from sklearn.ensemble import VotingRegressor
'details are omitted here. See classfication case below'
#### Voting classifier
from sklearn.ensemble import VotingClassifier
lr = LogisticRegression(random_state=SEED)
knn = KNN()
dt = DecisionTreeClassifier(random_state=SEED)
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]
vc = VotingClassifier(estimators=classifiers)
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)
accuracy_score(y_test, y_pred))

