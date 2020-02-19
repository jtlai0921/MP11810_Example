# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:36:45 2018

@author: Administrator
"""

from visdom import Visdom
import numpy as np
vis= Visdom()

# 顯示單一圖片
vis.image(
    np.random.rand(3, 256, 256),
    opts=dict(title='單一圖片', caption='圖片標題1'),
)

# 顯示網格圖片
vis.images(
    np.random.randn(20, 3, 64, 64),
    opts=dict(title='網格圖片', caption='圖片標題2')
)
