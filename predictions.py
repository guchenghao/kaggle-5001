#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/Desktop/FDA-5001/individualProject/predictions.py
# Project: /Users/guchenghao/Desktop/FDA-5001/individualProject
# Created Date: Sunday, October 28th 2018, 7:17:22 pm
# Author: Harold Gu
# -----
# Last Modified: Sunday, 28th October 2018 7:17:22 pm
# Modified By: Harold Gu
# -----
# Copyright (c) 2018 HKUST
# #
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###
import keras
from keras.layers import Dense, Input, Dropout
from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pandas as pd
import numpy as np


time_model = load_model('./model/model.h5')

time_model.load_weights('./model/best_weights.h5')

test_data = pd.read_csv('./model/test_new_data.csv')


id_pre = test_data['id']
test_data.drop(['id'], axis=1, inplace=True)


y_pred = time_model.predict(test_data, batch_size=100).reshape(-1, )

y_pred = np.log(y_pred)

y_pred = pd.Series(y_pred)

submission_csv = pd.concat([id_pre, y_pred], axis=1)

submission_csv.columns = ['Id', 'time']

submission_csv.to_csv('./submission_deep.csv', index=False)
