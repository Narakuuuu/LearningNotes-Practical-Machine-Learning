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

# 读数据集文件，输出样本量（行）和特征数（列）
data = pd.read_feather('data/house_sales.ftr')
data.shape

# 特征类型
data.head()

# 计算每列的缺失值数量。
null_sum = data.isnull().sum()
# 判断每列的缺失值数量是否少于总行数的 30%,根据上述条件筛选出需要保留的列名。
data.columns[null_sum < len(data) * 0.3] # colume will keep
# 判断每列的缺失值数量是否超过总行数的 30%,根据上述条件筛选出需要删除的列名,并删除
data.drop(columns=data.columns[null_sum > len(data) * 0.3], inplace=True)

# 特征（列）类型
data.dtypes