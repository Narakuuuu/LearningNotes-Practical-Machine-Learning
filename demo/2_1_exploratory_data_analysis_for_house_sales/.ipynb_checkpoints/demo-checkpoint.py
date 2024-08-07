# jupyter
# numpy: python里面做数据分析最常见的包
# pandas: 处理表的工具包
# matplotlib.pyplot/seaborn: 画图工具
# 数据处理+可视化
# -------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow

# 格式设置svg，清晰度更高
# -------------------------------------
from IPython import display
display.set_matplotlib_formats('svg')
# Alternative to set svg for newer versions
# import matplotlib_inline
# matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

data = pd.read_feather('data/house_sales.ftr')

data.shape

data.head()
