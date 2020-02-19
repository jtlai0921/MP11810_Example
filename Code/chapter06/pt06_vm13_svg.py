# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:43:59 2018

@author: Administrator
"""

from visdom import Visdom
import numpy as np

vis = Visdom()

svgstr = """
<svg height="300" width="300">
  <ellipse cx="80" cy="80" rx="50" ry="30"
   style="fill:red;stroke:purple;stroke-width:2" />
  抱歉，您的瀏覽器不支援線上顯示SVG物件！
</svg>
"""
vis.svg(
    svgstr=svgstr,
    opts=dict(title='SVG圖形')
)
