#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/Desktop/FDA-5001/individualProject/5001_deep_work.py
# Project: /Users/guchenghao/Desktop/FDA-5001/individualProject
# Created Date: Sunday, October 28th 2018, 6:32:50 pm
# Author: Harold Gu
# -----
# Last Modified: Sunday, 28th October 2018 6:32:52 pm
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
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, PolynomialFeatures
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import skew
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint


train_data = pd.read_csv('./train.csv')

train_data_2 = pd.read_csv('./combination_test.csv')

# train_data_3 = pd.read_csv('./train_2.csv')

train_data = pd.concat([train_data, train_data_2])

print(train_data)

test_data = pd.read_csv('./test.csv')

# test_data = test_data.iloc[[96, 30, 11, 8, 93,
#                                               65, 1, 41, 68, 5, 17, 72, 53, 43, 63, 82, 48, 88, 94, 80, 34, 33, 61, 86, 74, 78, 81, 10, 46, 40, 36, 77, 18, 27, 95, 85, 64, 75, 90, 45, 37, 6, 31, 54, 51, 13]]

Y = np.log(train_data['time'])
id_pre = test_data['id']

train_data.drop(['time'], axis=1, inplace=True)
train_data.drop(['id'], axis=1, inplace=True)
test_data.drop(['id'], axis=1, inplace=True)
# test_data.drop(['scale'], axis=1, inplace=True)
# train_data.drop(['scale'], axis=1, inplace=True)
# train_data.drop(['alpha'], axis=1, inplace=True)
# test_data.drop(['alpha'], axis=1, inplace=True)
# test_data.drop(['l1_ratio'], axis=1, inplace=True)
# train_data.drop(['l1_ratio'], axis=1, inplace=True)
# test_data.drop(['n_clusters_per_class'], axis=1, inplace=True)
# train_data.drop(['n_clusters_per_class'], axis=1, inplace=True)
train_data.drop(['random_state'], inplace=True, axis=1)
test_data.drop(['random_state'], inplace=True, axis=1)


print('训练数据集的维度: {0}'.format(train_data.shape))
print('测试数据集的维度: {0}'.format(test_data.shape))


# print('训练数据的描述统计量: \n {0}'.format(train_data.describe()))
print('训练数据的描述统计量: \n {0}'.format(test_data.describe()))


print('训练数据的属性信息: \n {0}'.format(train_data.info()))


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

train_data['l1_ratio'].loc[train_data['penalty'] != 3] = 0.0
train_data['alpha'].loc[train_data['penalty'] == 2] = 0.0
test_data['l1_ratio'].loc[test_data['penalty'] != 3] = 0.0
test_data['alpha'].loc[test_data['penalty'] == 2] = 0.0

train_data['n_jobs'].loc[train_data['n_jobs'] == -1] = 16
test_data['n_jobs'].loc[test_data['n_jobs'] == -1] = 16


train_data["n_samples_square"] = train_data["n_samples"] ** 2
test_data["n_samples_square"] = test_data["n_samples"] ** 2
# train_data["n_features_square"] = train_data["n_features"] ** 2
# test_data["n_features_square"] = test_data["n_features"] ** 2
train_data["max_iter_square"] = train_data["max_iter"] ** 2
test_data["max_iter_square"] = test_data["max_iter"] ** 2

train_data['data_amount'] = train_data['n_samples'].values * \
    train_data['n_features'].values
test_data['data_amount'] = test_data['n_samples'].values * \
    test_data['n_features'].values


# train_data['iter_multi_amount'] = train_data['n_samples'].values * \
#     train_data['max_iter'].values * train_data['n_features'].values
# test_data['iter_multi_amount'] = test_data['n_samples'].values * \
#     test_data['max_iter'].values * test_data['n_features'].values


train_data['n_clusters'] = train_data['n_classes'].values * \
    train_data['n_clusters_per_class'].values
test_data['n_clusters'] = test_data['n_classes'].values * \
    test_data['n_clusters_per_class'].values


train_data['amount_divide_classes_y'] = (train_data['data_amount'] /
                                         train_data['n_classes']) * train_data['flip_y']
test_data['amount_divide_classes_y'] = (test_data['data_amount'] /
                                        test_data['n_classes']) * test_data['flip_y']

# train_data['n_clusters'] = train_data['n_features'].values * \
#     train_data['max_iter'].values
# test_data['n_clusters'] = test_data['n_features'].values * \
#     test_data['max_iter'].values

# train_data["iter_multi_amount_square"] = train_data["iter_multi_amount"] ** 2
# test_data["iter_multi_amount_square"] = test_data["iter_multi_amount"] ** 2


# train_data['iter_multi_amount_divide_jobs'] = train_data['n_samples'].values * train_data['max_iter'].values * train_data['n_features'].values / \
#     train_data['n_jobs']
# test_data['iter_multi_amount_divide_jobs'] = test_data['n_samples'].values * test_data['max_iter'].values * test_data['n_features'].values / \
#     test_data['n_jobs']
# train_data["iter_multi_amount_divide_jobs_square"] = train_data["iter_multi_amount_divide_jobs"] ** 2
# test_data["iter_multi_amount_divide_jobs_square"] = test_data["iter_multi_amount_divide_jobs"] ** 2


train_data['iter_divide_jobs'] = train_data['max_iter'] / \
    train_data['n_jobs']
test_data['iter_divide_jobs'] = test_data['max_iter'] / \
    test_data['n_jobs']

train_data['samples_divide_jobs'] = train_data['n_samples'] / \
    train_data['n_jobs']
test_data['samples_divide_jobs'] = test_data['n_samples'] / \
    test_data['n_jobs']


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


robSc = StandardScaler()
train_data.loc[:, ['n_clusters', 'iter_divide_jobs', 'data_amount', 'scale', 'max_iter', 'n_samples', 'n_features', 'n_classes', 'n_informative', 'n_jobs', 'n_clusters_per_class', 'n_samples_square', 'samples_divide_jobs', 'n_samples_square', 'amount_divide_classes_y', 'max_iter_square']] = robSc.fit_transform(
    train_data.loc[:, ['n_clusters', 'iter_divide_jobs', 'data_amount', 'scale', 'max_iter', 'n_samples', 'n_features', 'n_classes', 'n_informative', 'n_jobs', 'n_clusters_per_class', 'n_samples_square', 'samples_divide_jobs', 'n_samples_square', 'amount_divide_classes_y', 'max_iter_square']])
test_data.loc[:, ['n_clusters', 'iter_divide_jobs', 'data_amount', 'scale', 'max_iter', 'n_samples', 'n_features', 'n_classes', 'n_informative', 'n_jobs', 'n_clusters_per_class', 'n_samples_square', 'samples_divide_jobs', 'n_samples_square', 'amount_divide_classes_y', 'max_iter_square']] = robSc.fit_transform(
    test_data.loc[:, ['n_clusters', 'iter_divide_jobs', 'data_amount', 'scale', 'max_iter', 'n_samples', 'n_features', 'n_classes', 'n_informative', 'n_jobs', 'n_clusters_per_class', 'n_samples_square', 'samples_divide_jobs', 'n_samples_square', 'amount_divide_classes_y', 'max_iter_square']])

print(train_data)

save_test_data = pd.concat([id_pre, test_data], axis=1)

save_test_data.to_csv('./model/test_new_data.csv', index=False)

# scores = []
# for i in range(1, 50):
#     rf_model = RandomForestRegressor(n_estimators=i)
#     rf_model.fit(train_data, Y)
#     prediction = rf_model.predict(train_data)
#     scores.append(mean_squared_error(Y, prediction))


# plt.plot(range(1, 50), scores)
# plt.show()

# X_train, X_test, Y_train, Y_test = train_test_split(
#     train_data, Y, test_size=0.2, random_state=77)

# X_train = train_data.iloc[0:400]
# Y_train = Y.iloc[0:400]

# X_test = train_data.iloc[400:500]
# Y_test = Y.iloc[400:500]


epochs = 50

inputs = Input(shape=(24, ))

dense_1 = Dense(128, activation='relu')(inputs)
dense_1 = Dropout(0.01)(dense_1)
# dense_1 = BatchNormalization()(dense_1)
dense_2 = Dense(128, activation='relu')(dense_1)
dense_2 = Dropout(0.01)(dense_2)

dense_3 = Dense(128, activation='relu')(dense_2)
dense_3 = Dropout(0.01)(dense_3)
# dense_3 = BatchNormalization()(dense_3)
# dense_4 = Dense(100, activation='relu')(dense_3)
# dense_4 = Dropout(0.5)(dense_4)
outputs = Dense(1, activation='linear')(dense_3)

time_model = Model(inputs=inputs, outputs=outputs)
time_model.compile(
    optimizer='adam', loss='mse')

# saveBestModel = ModelCheckpoint('./model/best_weights.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = time_model.fit(
    train_data, Y, epochs=epochs, batch_size=32, verbose=1)

time_model.summary()


time_model.save('./model/model.h5')
time_model.save_weights('./model/best_weights.h5')

scores = time_model.evaluate(train_data, Y, batch_size=32, verbose=1)
print(scores)

X_epoch = np.arange(0, epochs, 1)
plt.plot(X_epoch, history.history['loss'], label='acc')
# plt.plot(X_epoch, history.history['val_loss'], label='val_acc')
plt.legend(loc='best')
plt.show()
