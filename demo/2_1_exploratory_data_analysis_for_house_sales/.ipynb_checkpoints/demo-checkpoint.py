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

# 特征参数类型转换：'Sold Price', 'Listed Price', 'Tax assessed value', 'Annual tax amount' 类系转换为float
currency = ['Sold Price', 'Listed Price', 'Tax assessed value', 'Annual tax amount']
for c in currency:
    data[c] = (data[c]
               .replace(r'[$,-]', '', regex=True)       # 使用正则表达式将字符串中的 $、, 和 - 替换为空字符串。
               .replace(r'^\s*$', np.nan, regex=True)   # 使用正则表达式将仅包含空白字符的字符串替换为 NaN。
               .astype(float))    # 将列的数据类型转换为 float。


# 将面积单位转换为平方英尺
areas = ['Total interior livable area', 'Lot size']
for c in areas:
    # 检查是否包含 'Acres'
    acres = data[c].str.contains('Acres') == True

    # 移除单位标识符并将数据转换为浮点型
    col = data[c].replace(r'\b sqft\b|\b Acres\b|\b,\b','', regex=True).astype(float)

    # 将 'Acres' 转换为平方英尺
    col[acres] *= 43560

    # 更新数据
    data[c] = col

# 现在我们可以检查数值列的值。你会发现，几个列的最小值和最大值不合理。
data.describe()


# 条件筛选掉面积异常的房子数据
# 指示哪些行的面积小于 10 或大于 10,000（异常）。
abnormal = (data[areas[1]] < 10) | (data[areas[1]] > 1e4)
# 移除异常值
data = data[~abnormal]
# 计算异常值的数量
sum(abnormal)


# 售卖价格分布
# 绘制对数变换后的直方图
ax = sns.histplot(np.log10(data['Sold Price']))
# 设置 x 轴范围
ax.set_xlim([3, 8])
# 设置 x 轴刻度
ax.set_xticks(range(3, 9))
# 设置 x 轴刻度标签
ax.set_xticklabels(['%.0e'%a for a in 10**ax.get_xticks()])


# 获取房子类型特征中前20个最常见的值及其计数
data['Type'].value_counts()[0:20]


#不同类型房子的价格分布
# 选中房子类型
types = data['Type'].isin(['SingleFamily', 'Condo', 'MultiFamily', 'Townhouse'])
# 创建用于绘图的数据框，并绘制（displot） KDE 图
sns.displot(pd.DataFrame({'Sold Price':np.log10(data[types]['Sold Price']),
                          'Type':data[types]['Type']}),
            x='Sold Price', hue='Type', kind='kde')


# 不同类型房子每平米的售价分布
# 计算每平方英尺价格
data['Price per living sqft'] = data['Sold Price'] / data['Total interior livable area']
# 使用 Seaborn 的 boxplot 函数绘制箱线图，并设置 y 轴的范围为 0 到 2000。
ax = sns.boxplot(x='Type', y='Price per living sqft', data=data[types], fliersize=0)
ax.set_ylim([0, 2000])


# 创建一个箱线图，以展示不同邮政编码区域的每平方英尺价格分布
# 筛选出 Zip 列中出现次数最多的前 20 个邮政编码区域的数据
d = data[data['Zip'].isin(data['Zip'].value_counts()[:20].keys())]
# 使用 Seaborn 的 boxplot 函数绘制箱线图，并设置 y 轴的范围为 0 到 2000，同时将 x 轴标签旋转 90 度以便更好地显示。
ax = sns.boxplot(x='Zip', y='Price per living sqft', data=d, fliersize=0)
ax.set_ylim([0, 2000])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


# 展示多个列之间的相关性
# 选择需要计算相关性的列
columns = ['Sold Price', 'Listed Price', 'Annual tax amount', 'Price per living sqft', 'Elementary School Score', 'High School Score']
# 使用 Seaborn 的 heatmap 函数绘制热图，并设置图形的大小为 6x6，同时使用 RdYlGn 颜色映射，并在每个单元格中显示相关系数。
_, ax = plt.subplots(figsize=(6,6))
sns.heatmap(data[columns].corr(),annot=True,cmap='RdYlGn', ax=ax)