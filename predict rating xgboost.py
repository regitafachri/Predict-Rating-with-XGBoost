#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 10:19:03 2020

@author: regitafach
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf
from math import sqrt
plt.style.use('fivethirtyeight')

#import data
data = pd.read_csv('rating_top.csv',index_col=[0], parse_dates=[0])
data.head()

#Numerical-Categorical Split
x_num = data.drop(['GENRE','MOVIE'], axis=1)
label = ['GENRE','MOVIE']
x_cat = data[label]

#Missing value checking
x_num.isnull().any()
x_cat.isnull().any()

#Create new feature
x_num['date'] = x_num.index
x_num['DAYOFWEEK'] = x_num['date'].dt.dayofweek
x_num['QUARTER'] = x_num['date'].dt.quarter
x_num['MONTH'] = x_num['date'].dt.month
x_num['YEAR'] = x_num['date'].dt.year
x_num['DAYOFYEAR'] = x_num['date'].dt.dayofyear
x_num['DAYOFMONTH'] = x_num['date'].dt.day
x_num['WEEKOFYEAR'] = x_num['date'].dt.weekofyear

#categorical dummy
x_cat=pd.get_dummies(x_cat[label])

#Combine Categorical and Numerical Data
x_new = pd.concat([x_num, x_cat], axis=1)
x_new.head()

#Cek sebaran data
cek_sebaran = data.drop(['GENRE','MOVIE'], axis=1)
color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
_ = cek_sebaran.plot(style='.', figsize=(15,5), color=color_pal[0], title='Rating')

#Training testing split
split_date = '2019-09-01'
data_train = x_new.loc[x_new.index < split_date].copy()
data_test = x_new.loc[x_new.index >= split_date].copy()

x_train = data_train.drop(["Rating", "date"], axis = 1)
x_test = data_test.drop(["Rating", "date"], axis = 1)
y_train = data_train["Rating"]
y_test = data_test["Rating"]

#Create XGBoost Model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.utils import check_array

def bestparam_randCV(model,hyperparam,x_train, y_train, n_iter=1000):
    
    hyperparam = hyperparam
    randomizedCV = RandomizedSearchCV(model, param_distributions = hyperparam, cv = 10,
                                          n_iter = n_iter, scoring = 'neg_mean_squared_error', n_jobs=-1, 
                                          random_state = 42, verbose = True)
    randomizedCV.fit(x_train, y_train)
    
    #print (randomizedCV.cv_results_)
    print ('Best MSE', randomizedCV.score(x_train, y_train))
    print ('Best Param', randomizedCV.best_params_)
    return randomizedCV

reg         = XGBRegressor(n_estimators=1000)             

hyperparam = {'max_depth': [3,5,7,9],
              'min_child_weight': [1,3,5],
              'gamma': [0.0, 0.33333, 0.25, 0.5, 0.66667, 0.75],
              'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100], 
              'n_estimators': [100, 200, 300, 500, 750, 1000],
              'learning_rate': [0.01, 0.015, 0.02, 0.05, 0.08, 0.1],
} 
n_iter     = 10 
best_xgb1   = bestparam_randCV(reg, hyperparam, X_train, y_train, n_iter)

xgb1 = XGBRegressor(gamma            = best_xgb1.best_params_.get('gamma'),
                    reg_alpha        = best_xgb1.best_params_.get('reg_alpha'),
                    max_depth        = best_xgb1.best_params_.get('max_depth'),
                    n_estimators     = best_xgb1.best_params_.get('n_estimators'),
                    learning_rate    = best_xgb1.best_params_.get('learning_rate'),
                    min_child_weight = best_xgb1.best_params_.get('min_child_weight'))

result_xgb1 = xgb1.fit(X_train, y_train)

#Feature Importances
a = plot_importance(result_xgb1)
fig = a.figure
fig.set_size_inches(12, 15)

#Forecast on Test Set
data_test['RATING_Prediction'] = result_xgb1.predict(X_test)
data_all = pd.concat([data_test, data_train], sort=False)

_ = data_all[['Rating','RATING_Prediction']].plot(figsize=(15, 5))

#Error Metrics on Test Set
#RMSE
sqrt(mean_squared_error(y_true=data_test['Rating'],
                   y_pred=data_test['RATING_Prediction']))
#MAE
mean_absolute_error(y_true=data_test['Rating'],
                   y_pred=data_test['RATING_Prediction'])
#MAPE
mean_absolute_percentage_error(y_true=data_test['Rating'],
                   y_pred=data_test['RATING_Prediction'])


