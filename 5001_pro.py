#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/Desktop/FDA-5001/individualProject/5001_pro.py
# Project: /Users/guchenghao/Desktop/FDA-5001/individualProject
# Created Date: Thursday, October 25th 2018, 8:14:18 pm
# Author: Harold Gu
# -----
# Last Modified: Thursday, 25th October 2018 8:14:20 pm
# Modified By: Harold Gu
# -----
# Copyright (c) 2018 HKUST
# #
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import skew
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Input, Dropout
from keras import backend as K
from keras.models import Model


train_data = pd.read_csv('./train.csv')

train_data_2 = pd.read_csv('./combination_test.csv')

train_data = pd.concat([train_data, train_data_2])

# print(train_data)

test_data = pd.read_csv('./test.csv')

Y = np.log(train_data['time'])
id_pre = test_data['id']

train_data.drop(['time'], axis=1, inplace=True)
train_data.drop(['id'], axis=1, inplace=True)
test_data.drop(['id'], axis=1, inplace=True)
test_data.drop(['scale'], axis=1, inplace=True)
train_data.drop(['scale'], axis=1, inplace=True)
# test_data.drop(['n_clusters_per_class'], axis=1, inplace=True)
# train_data.drop(['n_clusters_per_class'], axis=1, inplace=True)
train_data.drop(['random_state'], inplace=True, axis=1)
test_data.drop(['random_state'], inplace=True, axis=1)


print('训练数据集的维度: {0}'.format(train_data.shape))
print('测试数据集的维度: {0}'.format(test_data.shape))


def l1(alpha, ratio, penalty):
    if penalty == 'l1':
        return(alpha)
    elif penalty == 'elasticnet':
        return(alpha * ratio)
    else:
        return(0)


def l2(alpha, ratio, penalty):
    if penalty == 'l2':
        return(alpha)
    elif penalty == 'elasticnet':
        return(alpha * (1-ratio))
    else:
        return(0)


train_data['l1'] = train_data.apply(lambda row: l1(
    row['alpha'], row['l1_ratio'], row['penalty']), axis=1)
train_data['l2'] = train_data.apply(lambda row: l2(
    row['alpha'], row['l1_ratio'], row['penalty']), axis=1)
test_data['l1'] = test_data.apply(lambda row: l1(
    row['alpha'], row['l1_ratio'], row['penalty']), axis=1)
test_data['l2'] = test_data.apply(lambda row: l2(
    row['alpha'], row['l1_ratio'], row['penalty']), axis=1)

train_data.drop(['alpha'], axis=1, inplace=True)
test_data.drop(['alpha'], axis=1, inplace=True)
test_data.drop(['l1_ratio'], axis=1, inplace=True)
train_data.drop(['l1_ratio'], axis=1, inplace=True)




# print('训练数据的描述统计量: \n {0}'.format(train_data.describe()))
print('训练数据的描述统计量: \n {0}'.format(test_data.describe()))


print('训练数据的属性信息: \n {0}'.format(train_data.info()))

numerical_features = train_data.select_dtypes(exclude=["object"]).columns


train_data['penalty'] = train_data['penalty'].replace({
    'l1': 0,
    'l2': 1,
    'none': 2,
    'elasticnet': 3
})

test_data['penalty'] = test_data['penalty'].replace({
    'l1': 0,
    'l2': 1,
    'none': 2,
    'elasticnet': 3
})

# train_data['l1_ratio'].loc[train_data['penalty'] != 3] = 0
# train_data['alpha'].loc[train_data['penalty'] == 2] = 0
# test_data['l1_ratio'].loc[test_data['penalty'] != 3] = 0
# test_data['alpha'].loc[test_data['penalty'] == 2] = 0

train_data['n_jobs'].loc[train_data['n_jobs'] == -1] = 16
test_data['n_jobs'].loc[test_data['n_jobs'] == -1] = 16

train_data["n_samples_square"] = train_data["n_samples"] ** 2
test_data["n_samples_square"] = test_data["n_samples"] ** 2
train_data["n_features_square"] = train_data["n_features"] ** 2
test_data["n_features_square"] = test_data["n_features"] ** 2
train_data["max_iter_square"] = train_data["max_iter"] ** 2
test_data["max_iter_square"] = test_data["max_iter"] ** 2
# train_data["flip_y_square"] = train_data["flip_y"] ** 2
# test_data["flip_y_square"] = test_data["flip_y"] ** 2
# train_data["n_classes_square"] = train_data["n_classes"] ** 2
# test_data["n_classes_square"] = test_data["n_classes"] ** 2

train_data['data_amount'] = train_data['n_samples'].values * \
    train_data['n_features'].values
test_data['data_amount'] = test_data['n_samples'].values * \
    test_data['n_features'].values


train_data['iter_multi_amount'] = train_data['n_samples'].values * \
    train_data['max_iter'].values * train_data['n_features'].values
test_data['iter_multi_amount'] = test_data['n_samples'].values * \
    test_data['max_iter'].values * test_data['n_features'].values


train_data['classed_multi_clusters'] = train_data['n_classes'].values * \
    train_data['n_clusters_per_class'].values
test_data['classed_multi_clusters'] = test_data['n_classes'].values * \
    test_data['n_clusters_per_class'].values

train_data['iter_multi_amount_divide_jobs'] = train_data['iter_multi_amount'] / \
    train_data['n_jobs']
test_data['iter_multi_amount_divide_jobs'] = test_data['iter_multi_amount'] / \
    test_data['n_jobs']
train_data["iter_multi_amount_divide_jobs_square"] = train_data["iter_multi_amount_divide_jobs"] ** 2
test_data["iter_multi_amount_divide_jobs_square"] = test_data["iter_multi_amount_divide_jobs"] ** 2


train_data['iter_divide_jobs'] = train_data['max_iter'] / \
    train_data['n_jobs']
test_data['iter_divide_jobs'] = test_data['max_iter'] / \
    test_data['n_jobs']

train_data['samples_divide_jobs'] = train_data['n_samples'] / \
    train_data['n_jobs']
test_data['samples_divide_jobs'] = test_data['n_samples'] / \
    test_data['n_jobs']

train_data['amount_divide_classes_y'] = (train_data['iter_multi_amount'] /
                                         train_data['n_classes']) * train_data['flip_y']
test_data['amount_divide_classes_y'] = (test_data['iter_multi_amount'] /
                                        test_data['n_classes']) * test_data['flip_y']


train_data['features_divide_jobs'] = train_data['n_features'] / \
    train_data['n_jobs']
test_data['features_divide_jobs'] = test_data['n_features'] / \
    test_data['n_jobs']


train_data['n_clusters'] = train_data['n_features'].values * \
    train_data['max_iter'].values
test_data['n_clusters'] = test_data['n_features'].values * \
    test_data['max_iter'].values

train_data['samples_divide_jobs'] = train_data['n_samples'] / \
    train_data['n_jobs']
test_data['samples_divide_jobs'] = test_data['n_samples'] / \
    test_data['n_jobs']


# print(test_data['samples_divide_jobs'])

# test_data['n_jobs'] = test_data['n_jobs'].replace({
#     -1: 16
# })

# train_data['n_jobs'] = train_data['n_jobs'].replace({
#     -1: 16
# })


print(train_data['n_jobs'])


plot_train_data = pd.concat([train_data, Y], axis=1)
corrmat = plot_train_data.corr()  # ! 输出DataFrame的相关系数矩阵, (38, 38)
print(corrmat.time)


train_penalty = pd.get_dummies(train_data['penalty'])
test_penalty = pd.get_dummies(test_data['penalty'])
train_data.drop(['penalty'], axis=1, inplace=True)
test_data.drop(['penalty'], axis=1, inplace=True)

train_data = pd.concat([train_penalty, train_data], axis=1)
test_data = pd.concat([test_penalty, test_data], axis=1)


robSc = MinMaxScaler()
train_data.loc[:, ['samples_divide_jobs', 'n_clusters', 'features_divide_jobs', 'n_features_square', 'max_iter_square', 'n_samples_square', 'max_iter', 'n_samples', 'n_features', 'n_classes', 'n_informative', 'iter_multi_amount', 'samples_divide_jobs', 'iter_divide_jobs', 'amount_divide_classes_y', 'data_amount', 'iter_multi_amount_divide_jobs_square']] = robSc.fit_transform(
    train_data.loc[:, ['samples_divide_jobs', 'n_clusters', 'features_divide_jobs', 'n_features_square', 'max_iter_square', 'n_samples_square', 'max_iter', 'n_samples', 'n_features', 'n_classes', 'n_informative', 'iter_multi_amount', 'samples_divide_jobs', 'iter_divide_jobs', 'amount_divide_classes_y', 'data_amount', 'iter_multi_amount_divide_jobs_square']])
test_data.loc[:, ['samples_divide_jobs', 'n_clusters', 'features_divide_jobs', 'n_features_square', 'max_iter_square', 'n_samples_square', 'max_iter', 'n_samples', 'n_features', 'n_classes', 'n_informative', 'iter_multi_amount', 'samples_divide_jobs', 'iter_divide_jobs', 'amount_divide_classes_y', 'data_amount', 'iter_multi_amount_divide_jobs_square']] = robSc.fit_transform(
    test_data.loc[:, ['samples_divide_jobs', 'n_clusters', 'features_divide_jobs', 'n_features_square', 'max_iter_square', 'n_samples_square', 'max_iter', 'n_samples', 'n_features', 'n_classes', 'n_informative', 'iter_multi_amount', 'samples_divide_jobs', 'iter_divide_jobs', 'amount_divide_classes_y', 'data_amount', 'iter_multi_amount_divide_jobs_square']])


print(train_data)

# scores = []
# for i in range(1, 50):
#     rf_model = RandomForestRegressor(n_estimators=i)
#     rf_model.fit(train_data, Y)
#     prediction = rf_model.predict(train_data)
#     scores.append(mean_squared_error(Y, prediction))


# plt.plot(range(1, 50), scores)
# plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(
    train_data, Y, test_size=0.2, random_state=66)


# gbdt_model = GradientBoostingRegressor(learning_rate=0.1,n_estimators=150)
xgb_model_1 = xgb.XGBRegressor(
    learning_rate=0.07, n_estimators=150, max_depth=4, reg_lambda=0.3, reg_alpha=0.2)
xgb_model_2 = xgb.XGBRegressor(
    learning_rate=0.07, n_estimators=160, max_depth=4, reg_lambda=0.3, reg_alpha=0.2)
xgb_model_3 = xgb.XGBRegressor(
    learning_rate=0.07, n_estimators=140, max_depth=4, reg_lambda=0.3, reg_alpha=0.2)
xgb_model_4 = xgb.XGBRegressor(
    learning_rate=0.07, n_estimators=130, max_depth=4, reg_lambda=0.3, reg_alpha=0.2)
xgb_model_5 = xgb.XGBRegressor(
    learning_rate=0.07, n_estimators=180, max_depth=4, reg_lambda=0.3, reg_alpha=0.2)
xgb_model_6 = xgb.XGBRegressor(
    learning_rate=0.07, n_estimators=190, max_depth=4, reg_lambda=0.3, reg_alpha=0.2)
xgb_model_7 = xgb.XGBRegressor(
    learning_rate=0.07, n_estimators=300, max_depth=4, reg_lambda=0.3, reg_alpha=0.2)
xgb_model_8 = xgb.XGBRegressor(
    learning_rate=0.07, n_estimators=420, max_depth=4, reg_lambda=0.3, reg_alpha=0.2)
xgb_model_9 = xgb.XGBRegressor(
    learning_rate=0.07, n_estimators=520, max_depth=4, reg_lambda=0.3, reg_alpha=0.2)
xgb_model_10 = xgb.XGBRegressor(
    learning_rate=0.07, n_estimators=560, max_depth=4, reg_lambda=0.3, reg_alpha=0.2)
xgb_model_11 = xgb.XGBRegressor(
    learning_rate=0.07, n_estimators=600, max_depth=4, reg_lambda=0.3, reg_alpha=0.2)
mlp_model = MLPRegressor(hidden_layer_sizes=(
    100, 100, 100, 100), learning_rate='adaptive')
# xgb_model_4 = xgb.XGBRegressor(learning_rate=0.08, n_estimators=130, max_depth=4, reg_lambda=0.3, reg_alpha=0.2, random_state=64)
# rf_model = RandomForestRegressor(n_estimators=50)
# ada_model = AdaBoostRegressor(n_estimators=80, learning_rate=0.05)

# gbdt_model.fit(X_train, Y_train)
xgb_model_1.fit(X_train, Y_train)
xgb_model_2.fit(X_train, Y_train)
xgb_model_3.fit(X_train, Y_train)
xgb_model_4.fit(X_train, Y_train)
xgb_model_5.fit(X_train, Y_train)
xgb_model_6.fit(X_train, Y_train)
xgb_model_7.fit(X_train, Y_train)
xgb_model_8.fit(X_train, Y_train)
xgb_model_9.fit(X_train, Y_train)
xgb_model_10.fit(X_train, Y_train)
xgb_model_11.fit(X_train, Y_train)
mlp_model.fit(X_train, Y_train)
# mlp_model.fit(X_train, Y_train)
# mlp_model.fit(X_train, Y_train)
# svm_model.fit(X_train, Y_train)
# rf_model.fit(X_train, Y_train)
# ada_model.fit(train_data, Y)


y_pred_xgb_1 = xgb_model_1.predict(test_data)
y_pred_xgb_2 = xgb_model_2.predict(test_data)
y_pred_xgb_3 = xgb_model_3.predict(test_data)
y_pred_xgb_4 = xgb_model_4.predict(test_data)
y_pred_xgb_5 = xgb_model_5.predict(test_data)
y_pred_xgb_6 = xgb_model_6.predict(test_data)
y_pred_xgb_7 = xgb_model_7.predict(test_data)
y_pred_xgb_8 = xgb_model_8.predict(test_data)
y_pred_xgb_9 = xgb_model_9.predict(test_data)
y_pred_xgb_10 = xgb_model_10.predict(test_data)
y_pred_xgb_11 = xgb_model_11.predict(test_data)
mlp_model_y_pred = mlp_model.predict(test_data)
# y_pred_rf = rf_model.predict(test_data)
prediction_xgb = xgb_model_1.predict(X_test)
prediction_xgb_1 = xgb_model_2.predict(X_test)
prediction_xgb_2 = xgb_model_3.predict(X_test)
prediction_xgb_3 = xgb_model_4.predict(X_test)
prediction_xgb_4 = xgb_model_5.predict(X_test)
prediction_xgb_5 = xgb_model_6.predict(X_test)
prediction_xgb_6 = xgb_model_7.predict(X_test)
prediction_xgb_7 = xgb_model_8.predict(X_test)
prediction_xgb_8 = xgb_model_9.predict(X_test)
prediction_xgb_9 = xgb_model_10.predict(X_test)
prediction_xgb_10 = xgb_model_11.predict(X_test)
prediction_mlp = mlp_model.predict(X_test)
# prediction_mlp = mlp_model.predict(X_test)
# prediction_mlp= mlp_model.predict(X_test)
# prediction_SVM = svm_model.predict(X_test)

prediction_all = (prediction_xgb + prediction_xgb_1 +
                  prediction_xgb_2 + prediction_xgb_3 + prediction_xgb_4 + prediction_xgb_5 + prediction_xgb_6 + prediction_xgb_7 + prediction_xgb_8 + prediction_xgb_9 + prediction_xgb_10) / 11

y_pred = np.exp((y_pred_xgb_1 + y_pred_xgb_2 + y_pred_xgb_3 + y_pred_xgb_4 + y_pred_xgb_5 + y_pred_xgb_6 + y_pred_xgb_7 + y_pred_xgb_8 + y_pred_xgb_9 + y_pred_xgb_10 + y_pred_xgb_11) / 11)

# y_pred = np.exp(y_pred)

print(mean_squared_error(Y_test, prediction_all))
print(mean_squared_error(Y_test, prediction_mlp))

print(Y.median())


y_pred = pd.Series(y_pred)

submission_csv = pd.concat([id_pre, y_pred], axis=1)

submission_csv.columns = ['Id', 'time']

submission_csv.to_csv('./submission.csv', index=False)
