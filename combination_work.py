#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/Desktop/FDA-5001/individualProject/combination_work.py
# Project: /Users/guchenghao/Desktop/FDA-5001/individualProject
# Created Date: Tuesday, October 30th 2018, 4:11:19 pm
# Author: Harold Gu
# -----
# Last Modified: Tuesday, 30th October 2018 4:11:19 pm
# Modified By: Harold Gu
# -----
# Copyright (c) 2018 HKUST
# #
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###
import pandas as pd
import numpy as np


deep_work = pd.read_csv('./submission_deep.csv')

traditional_work = pd.read_csv('./submission.csv')
base = pd.read_csv('./submission_final.csv')
new_base = pd.read_csv('./11.20.csv')

test_data = pd.read_csv('./test.csv')


y_pred = (0.9 * traditional_work['time'] + 0.1 * deep_work['time'])


submission_csv = pd.concat([deep_work['Id'], y_pred], axis=1)

submission_csv.sort_values('Id', inplace=True)


# final_pred = pd.Series(final_pred)

# submission_csv = pd.concat([id_pre, final_pred], axis=1)

# submission_csv.columns = ['Id', 'time']

submission_csv.to_csv('./submission_final.csv', index=False)

combination_test = pd.concat([test_data, new_base['time']], axis=1)

combination_test.to_csv('./combination_test.csv', index=False)
