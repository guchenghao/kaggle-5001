#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/Desktop/FDA-5001/individualProject/combination_test.py
# Project: /Users/guchenghao/Desktop/FDA-5001/individualProject
# Created Date: Tuesday, October 30th 2018, 10:58:48 pm
# Author: Harold Gu
# -----
# Last Modified: Tuesday, 30th October 2018 10:58:49 pm
# Modified By: Harold Gu
# -----
# Copyright (c) 2018 HKUST
# #
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###
import pandas as pd
from sklearn.metrics import mean_squared_error


last = pd.read_csv('./submission_deep_last.csv')
# now = pd.read_csv('./submission_deep.csv')
sub = pd.read_csv('./submission.csv')
final = pd.read_csv('./submission_deep.csv')
base = pd.read_csv('./submission_averaging_on_best3.csv')


# print(mean_squared_error(last['time'], sub['time']))
# print(mean_squared_error(last['time'], now['time']))
# print(mean_squared_error(last['time'], final['time']))
print(mean_squared_error(last['time'], final['time']))
print(mean_squared_error(base['time'], sub['time']))
