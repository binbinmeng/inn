#!/usr/bin/env python
# coding=utf-8


import numpy as np
z = np.array([-6.4, -1.2, -3.2, 10.6, -8.8, 13.6, -7.6, -1.2, 2.6, 1.4])
z = np.array([])

accu = np.exp(z) / sum(np.exp(z))

for i in range(10):
    print("{}: {}".format(i, accu[i]))