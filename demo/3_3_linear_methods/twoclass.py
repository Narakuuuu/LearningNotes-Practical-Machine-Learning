# 【线性回归解决分类问题】【0/1分类】使用线性回归模型来预测学生是否通过考试

# python里面做数据分析最常见的包
import numpy as np

# 处理表的工具包
import pandas as pd

# 创建数据集；创建了一个字典 data，包含两个键：学习时间 和 是否通过考试，分别对应自变量和因变量的数据。
data = {'学习时间': [1, 2, 3, 4, 5],
        '是否通过考试': [0, 0, 0, 1, 1]}
# 使用 pandas 的 DataFrame 将字典转换为一个数据框 df。
df = pd.DataFrame(data)

# 来自 scikit-learn 的线性回归模型，用于进行回归分析。
from sklearn.linear_model import LinearRegression

# 自变量和因变量
# X 是自变量（特征），即 学习时间 列。
X = df[['学习时间']]
# y 是因变量（目标），即 是否通过考试 列。
y = df['是否通过考试']

# 创建线性回归模型
model = LinearRegression()
# 使用fit方法去拟合模型，X 是自变量，y 是因变量。
model.fit(X, y)

# 打印模型参数、
# model.intercept_：模型的截距（偏置项）
print(f'截距: {model.intercept_}')
# model.coef_：模型的系数（权重），对应自变量 学习时间
print(f'系数: {model.coef_}')
# 函数表达为 y = model.coef_ * X + model.intercept_

# 预测
predictions = model.predict(X)
print(predictions)

# 应用阈值
# 定义一个阈值 threshold 为 0.5
threshold = 0.5
# 使用列表推导式，根据预测值 predictions 和阈值 threshold 生成分类结果 predicted_classes：
# 如果预测值大于阈值，则分类为 1（通过考试）。
# 否则分类为 0（未通过考试）。
predicted_classes = [1 if pred > threshold else 0 for pred in predictions]
print(predicted_classes)

# 好的，让我们通过一个具体的案例来讲解如何使用线性回归来处理分类问题。尽管线性回归不是分类问题的最佳选择，但通过这个案例，你可以更好地理解其应用和局限性。
#
# ### 案例：预测学生是否通过考试
#
# 假设我们有一个数据集，其中包含学生的学习时间（小时）和他们是否通过考试的结果（通过记为1，未通过记为0）。我们的目标是根据学习时间预测学生是否通过考试。
#
# #### 数据集
#
# | 学习时间（小时） | 是否通过考试（0或1） |
# |------------------|-----------------------|
# | 1                | 0                     |
# | 2                | 0                     |
# | 3                | 0                     |
# | 4                | 1                     |
# | 5                | 1                     |
#
# ### 步骤
#
# #### 1. 数据准备
#
# 首先，我们将数据分为自变量（学习时间）和因变量（是否通过考试）。
#
# ```python
# import numpy as np
# import pandas as pd
#
# # 创建数据集
# data = {'学习时间': [1, 2, 3, 4, 5],
#         '是否通过考试': [0, 0, 0, 1, 1]}
# df = pd.DataFrame(data)
# ```
#
# #### 2. 拟合线性回归模型
#
# 使用线性回归模型来拟合数据。
#
# ```python
# from sklearn.linear_model import LinearRegression
#
# # 自变量和因变量
# X = df[['学习时间']]
# y = df['是否通过考试']
#
# # 创建线性回归模型
# model = LinearRegression()
# model.fit(X, y)
#
# # 打印模型参数
# print(f'截距: {model.intercept_}')
# print(f'系数: {model.coef_}')
# ```
#
# 假设得到的模型参数为：
# \[ \text{是否通过} = -1 + 0.4 \times \text{学习时间} \]
#
# #### 3. 预测
#
# 使用拟合好的模型进行预测。
#
# ```python
# # 预测
# predictions = model.predict(X)
# print(predictions)
# ```
#
# 预测值为：
# \[ [-0.6, -0.2, 0.2, 0.6, 1.0] \]
#
# #### 4. 应用阈值
#
# 选择阈值0.5，将预测值映射到类别标签。
#
# ```python
# # 应用阈值
# threshold = 0.5
# predicted_classes = [1 if pred > threshold else 0 for pred in predictions]
# print(predicted_classes)
# ```
#
# 得到的类别预测为：
# \[ [0, 0, 0, 1, 1] \]
#
# ### 结果分析
#
# 通过以上步骤，我们使用线性回归模型对学生是否通过考试进行了预测，并应用了阈值将连续预测值转换为类别标签。结果表明，模型能够正确分类大部分样本，但由于线性回归的输出是连续值，可能会出现一些不合理的预测值（如负值），这在分类问题中是不合适的。
#
# ### 线性回归用于分类的局限性
#
# 1. **输出范围不合适**：线性回归的输出可以是任何实数，而分类问题通常需要一个有限的类别集合。
# 2. **误差度量不匹配**：分类问题通常使用分类精度、混淆矩阵、AUC-ROC等指标来评估模型，而线性回归使用均方误差（MSE）或均方根误差（RMSE）等指标，这些指标不适合分类问题。
# 3. **模型假设不匹配**：线性回归假设自变量和因变量之间存在线性关系，而分类问题的类别标签可能与自变量之间的关系更加复杂。
#
# ### 更好的选择：逻辑回归
#
# 逻辑回归是一种专门用于二分类问题的模型，通常效果更好。它将线性回归的输出通过逻辑函数（Sigmoid函数）映射到0到1之间的概率值，然后根据概率值进行分类。
#
# ### 总结
#
# 虽然线性回归可以通过阈值法进行调整来处理分类问题，但效果通常不如专门的分类模型（如逻辑回归）好。对于分类问题，建议使用逻辑回归、支持向量机、决策树等专门的分类算法。

