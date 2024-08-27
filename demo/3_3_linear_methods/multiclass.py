# 【线性回归解决分类问题】【多类别分类】使用线性回归模型来预测花的种类

# python里面做数据分析最常见的包
import numpy as np
# 处理表的工具包
import pandas as pd

# 创建数据集：创建一个字典data，包含三个键'花瓣长度'，'花瓣宽度'，'花的种类'，，分别对应自变量和因变量的数据。
data = {'花瓣长度': [1.4, 1.3, 4.7, 4.5, 5.1, 5.9],
        '花瓣宽度': [0.2, 0.2, 1.4, 1.5, 1.8, 2.1],
        '花的种类': ['Setosa', 'Setosa', 'Versicolor', 'Versicolor', 'Virginica', 'Virginica']}
# 使用 pandas 的 DataFrame 将字典转换为一个数据框 df。
df = pd.DataFrame(data)

# 来自 scikit-learn 的标签解码，用于把类别转化为数值类型
from sklearn.preprocessing import LabelEncoder

# 编码因变量
label_encoder = LabelEncoder()
# 将花的种类编码为数值类型（例如，Setosa编码为0，Versicolor编码为1，Virginica编码为2）。
df['花的种类编码'] = label_encoder.fit_transform(df['花的种类'])


# 来自 scikit-learn 的线性回归模型，用于进行回归分析。
from sklearn.linear_model import LinearRegression

# 自变量和因变量
# X 是自变量（特征）
X = df[['花瓣长度', '花瓣宽度']]
# y 是因变量（目标）
y = df['花的种类编码']

# 创建线性回归模型
model = LinearRegression()
# 使用fit方法去拟合模型，X 是自变量，y 是因变量。
model.fit(X, y)

# 打印模型参数
print(f'截距: {model.intercept_}')
print(f'系数: {model.coef_}')
# 函数表达为 y = <model.coef_ , X> + model.intercept_


# 预测
predictions = model.predict(X)
print(predictions)

# 应用四舍五入
predicted_classes = np.round(predictions).astype(int)
print(predicted_classes)


# 线性回归本质上是用于回归问题的，但我们可以通过一些调整将其应用于多类分类问题。不过，这种方法通常不如专门为分类问题设计的算法（如逻辑回归、支持向量机、决策树等）效果好。下面我们通过一个具体的案例来讲解如何使用线性回归处理多类分类问题。
#
# ### 案例：预测花的种类
#
# 假设我们有一个数据集，包含花的特征（如花瓣长度、花瓣宽度等）和花的种类（例如，三种不同的花：Setosa、Versicolor和Virginica）。我们的目标是根据花的特征预测花的种类。
#
# #### 数据集
#
# | 花瓣长度 | 花瓣宽度 | 花的种类      |
# |----------|----------|---------------|
# | 1.4      | 0.2      | Setosa        |
# | 1.3      | 0.2      | Setosa        |
# | 4.7      | 1.4      | Versicolor    |
# | 4.5      | 1.5      | Versicolor    |
# | 5.1      | 1.8      | Virginica     |
# | 5.9      | 2.1      | Virginica     |
#
# ### 步骤
#
# #### 1. 数据准备
#
# 首先，我们将数据分为自变量（花瓣长度和花瓣宽度）和因变量（花的种类）。
#
# ```python
# import numpy as np
# import pandas as pd
#
# # 创建数据集
# data = {'花瓣长度': [1.4, 1.3, 4.7, 4.5, 5.1, 5.9],
#         '花瓣宽度': [0.2, 0.2, 1.4, 1.5, 1.8, 2.1],
#         '花的种类': ['Setosa', 'Setosa', 'Versicolor', 'Versicolor', 'Virginica', 'Virginica']}
# df = pd.DataFrame(data)
# ```
#
# #### 2. 编码因变量
#
# 将花的种类编码为数值类型（例如，Setosa编码为0，Versicolor编码为1，Virginica编码为2）。
#
# ```python
# from sklearn.preprocessing import LabelEncoder
#
# # 编码因变量
# label_encoder = LabelEncoder()
# df['花的种类编码'] = label_encoder.fit_transform(df['花的种类'])
# ```
#
# #### 3. 拟合线性回归模型
#
# 使用线性回归模型来拟合数据。
#
# ```python
# from sklearn.linear_model import LinearRegression
#
# # 自变量和因变量
# X = df[['花瓣长度', '花瓣宽度']]
# y = df['花的种类编码']
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
# #### 4. 预测
#
# 使用拟合好的模型进行预测。
#
# ```python
# # 预测
# predictions = model.predict(X)
# print(predictions)
# ```
#
# 预测值可能为：
# \[ [0.1, 0.2, 1.3, 1.4, 2.0, 2.1] \]
#
# #### 5. 应用阈值
#
# 将预测值四舍五入到最近的整数，以得到类别标签。
#
# ```python
# # 应用四舍五入
# predicted_classes = np.round(predictions).astype(int)
# print(predicted_classes)
# ```
#
# 得到的类别预测可能为：
# \[ [0, 0, 1, 1, 2, 2] \]
#
# ### 结果分析
#
# 通过以上步骤，我们使用线性回归模型对花的种类进行了预测，并将连续预测值四舍五入到最近的整数，以得到类别标签。结果表明，模型能够正确分类大部分样本，但由于线性回归的输出是连续值，可能会出现一些不合理的预测值。
#
# ### 线性回归用于多类分类的局限性
#
# 1. **输出范围不合适**：线性回归的输出可以是任何实数，而多类分类问题通常需要一个有限的类别集合。
# 2. **误差度量不匹配**：分类问题通常使用分类精度、混淆矩阵、AUC-ROC等指标来评估模型，而线性回归使用均方误差（MSE）或均方根误差（RMSE）等指标，这些指标不适合分类问题。
# 3. **模型假设不匹配**：线性回归假设自变量和因变量之间存在线性关系，而分类问题的类别标签可能与自变量之间的关系更加复杂。
#
# ### 更好的选择：逻辑回归和其他分类算法
#
# 对于多类分类问题，建议使用多类逻辑回归（如One-vs-Rest或Softmax回归）、支持向量机、决策树、随机森林等专门的分类算法。
#
# ### 总结
#
# 虽然线性回归可以通过四舍五入等方法进行调整来处理多类分类问题，但效果通常不如专门的分类模型好。对于多类分类问题，建议使用多类逻辑回归、支持向量机、决策树等专门的分类算法。
#
