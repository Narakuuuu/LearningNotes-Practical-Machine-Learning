import numpy as np
import pandas as pd

# 创建数据集
data = {'花瓣长度': [1.4, 1.3, 4.7, 4.5, 5.1, 5.9],
        '花瓣宽度': [0.2, 0.2, 1.4, 1.5, 1.8, 2.1],
        '花的种类': ['Setosa', 'Setosa', 'Versicolor', 'Versicolor', 'Virginica', 'Virginica']}
df = pd.DataFrame(data)

from sklearn.preprocessing import LabelEncoder

# 编码因变量
label_encoder = LabelEncoder()
df['花的种类编码'] = label_encoder.fit_transform(df['花的种类'])

# LogisticRegression 用于创建逻辑回归模型。
from sklearn.linear_model import LogisticRegression

# 自变量和因变量
X = df[['花瓣长度', '花瓣宽度']]
y = df['花的种类编码']

# 创建Softmax回归模型
# 创建一个 LogisticRegression 实例 model，并指定 multi_class='multinomial' 以使用 Softmax 回归，solver='lbfgs' 作为优化算法。
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
# 使用fit方法去拟合（训练）模型，X 是自变量，y 是因变量。
model.fit(X, y)

# 预测并打印概率
probabilities = model.predict_proba(X)
print(probabilities)

# # Softmax Regression Demo
# 好的，Softmax回归（也称为多类逻辑回归）是一种专门用于多类分类问题的模型。它的输出是每个类别的概率，且这些概率的和为1。下面我们通过一个具体的案例来讲解如何使用Softmax回归来处理多类分类问题。
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
# #### 3. 拟合Softmax回归模型
#
# 使用Softmax回归模型来拟合数据。
#
# ```python
# from sklearn.linear_model import LogisticRegression
#
# # 自变量和因变量
# X = df[['花瓣长度', '花瓣宽度']]
# y = df['花的种类编码']
#
# # 创建Softmax回归模型
# model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
# model.fit(X, y)
# ```
#
# #### 4. 预测概率
#
# 使用拟合好的模型进行预测，并输出每个类别的概率。
#
# ```python
# # 预测概率
# probabilities = model.predict_proba(X)
# print(probabilities)
# ```
#
# ### 结果分析
#
# 通过以上步骤，我们使用Softmax回归模型对花的种类进行了预测，并输出了每个类别的概率。结果表明，模型能够给出每个样本属于每个类别的概率。
#
# #### 示例输出
#
# 假设我们有以下预测概率：
#
# | 样本索引 | Setosa概率 | Versicolor概率 | Virginica概率 |
# |----------|------------|----------------|---------------|
# | 0        | 0.95       | 0.03           | 0.02          |
# | 1        | 0.94       | 0.04           | 0.02          |
# | 2        | 0.02       | 0.85           | 0.13          |
# | 3        | 0.03       | 0.80           | 0.17          |
# | 4        | 0.01       | 0.10           | 0.89          |
# | 5        | 0.01       | 0.05           | 0.94          |
#
# 这些概率表示每个样本属于每个类别的可能性。
#
# ### Softmax回归的优势
#
# 1. **概率输出**：Softmax回归直接输出每个类别的概率，且这些概率的和为1。
# 2. **适用于多类分类**：Softmax回归专门设计用于多类分类问题，效果通常优于线性回归等回归模型。
# 3. **模型简洁**：Softmax回归模型相对简单，易于理解和实现。
#
# ### 总结
#
# Softmax回归是一种有效的多类分类算法，能够直接输出每个类别的概率。相比于线性回归等回归模型，Softmax回归在处理多类分类问题时效果更好。


# # Softmax Regression 原理
#