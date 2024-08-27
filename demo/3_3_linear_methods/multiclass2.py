# 【线性回归解决分类问题】【多类别分类】使用线性回归模型来预测花属于每一类的概率

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

from sklearn.preprocessing import OneHotEncoder

# 创建One-Hot编码
# One-Hot编码：编码是一种将类别变量转换为数值表示的方法
# One-Hot 编码的步骤：
#   类别识别：识别所有可能的类别。
#   向量化：为每个类别分配一个唯一的二进制向量。
# 原始数据
# |花的种类|
# |Setosa|
# |Versicolor|
# |Virginica|
# |Setosa|
# One-Hot 编码后
# |Setosa	|Versicolor	|Virginica
# |1	    |0	        |0
# |0	    |1	        |0
# |0	    |0	        |1
# |1	    |0	        |0
onehot_encoder = OneHotEncoder(sparse_output=False)
y_onehot = onehot_encoder.fit_transform(df[['花的种类编码']])
print(f'y_onehot ： \n{y_onehot}')

from sklearn.linear_model import LinearRegression

# 自变量
X = df[['花瓣长度', '花瓣宽度']]

# 创建线性回归模型列表,为每一个类型（列）创建线性回归模型
models = []
for i in range(y_onehot.shape[1]):
    model = LinearRegression()
    print(f'y_onehot{i} ： \n{y_onehot[:, i]}')
    # 这个拟合有点东西，之前的case中（如twoclass.py）fit拟合的目标值是一个具体的数值，
    # 本case中y_onehot[:, i]的值是一个向量，这里其实有一层转化关系，y_onehot每一列
    # 代表一个类别。
    # Setosa、Versicolor 和 Virginica One-Hot 编码之后的表达为：
    #   Setosa     -> [1, 0, 0]
    #   Versicolor -> [0, 1, 0]
    #   Virginica  -> [0, 0, 1]
    # y_onehot的输出为：
    #   [[1. 0. 0.]
    #   [1. 0. 0.]
    #   [0. 1. 0.]
    #   [0. 1. 0.]
    #   [0. 0. 1.]
    #   [0. 0. 1.]]
    # 映射关系为：
    # PetalLength | PetalWidth | Species
    # ------------|------------|---------
    # 1.4         | 0.2        | Setosa
    # 1.3         | 0.2        | Setosa
    # 4.5         | 1.5        | Versicolor
    # 4.9         | 1.5        | Versicolor
    # 6.1         | 2.3        | Virginica
    # 5.9         | 2.1        | Virginica
    # -----------------------------------------
    # PetalLength | PetalWidth | Setosa | Versicolor | Virginica
    # ------------|------------|--------|------------|----------
    # 1.4         | 0.2        | 1      | 0          | 0
    # 1.3         | 0.2        | 1      | 0          | 0
    # 4.5         | 1.5        | 0      | 1          | 0
    # 4.9         | 1.5        | 0      | 1          | 0
    # 6.1         | 2.3        | 0      | 0          | 1
    # 5.9         | 2.1        | 0      | 0          | 1
    # 每一列代表一个类别，如Setosa这一列，表示的是是否属于Setosa类，他值只有0/1两种，1属于Setosa，0则不属于
    model.fit(X, y_onehot[:, i])
    models.append(model)

print('model in models')
for index,model in enumerate(models):
    # 打印模型参数
    print(f'index: {index}')
    print(f'截距: {model.intercept_}')
    print(f'系数: {model.coef_}')

# 预测：对每个模型进行预测，得到预测值 predictions
# 进行预测并打印每个模型的预测结果
predictions_list = []
for index, model in enumerate(models):
    # 这里神奇的是predict的结果也是一个向量（列），每一个样本都会在该模型上预测一次。对应的表为
    # PetalLength | PetalWidth | Setosa   | Versicolor | Virginica
    # ------------|------------|----------|------------|----------
    # 1.4         | 0.2        | 0.941780 | 0.173932   | 0.000000
    # 1.3         | 0.2        | 0.994265 | 0.000437   | 0.005297
    # 4.5         | 1.5        | 0.001701 | 1.000000   | 0.000000
    # 4.9         | 1.5        | 0.172669 | 0.308864   | 0.518467
    # 6.1         | 2.3        | 0.055742 | 0.139816   | 0.804442
    # 5.9         | 2.1        | 0.000000 | 0.317758   | 0.848399
    prediction = model.predict(X)
    predictions_list.append(prediction)
    print(f"Model {index} predictions: {prediction}")
# 将预测结果转置
predictions = np.array(predictions_list).T

# 将输出转换为概率：使用 np.clip 将预测值限制在 0 到 1 之间。
probabilities = np.clip(predictions, 0, 1)
print(probabilities)

# 将预测值归一化为概率，使每行的和为 1。
probabilities /= probabilities.sum(axis=1, keepdims=True)
print(probabilities)


# 将概率转化为类别 数值
predicted_class_indices = np.argmax(probabilities, axis=1)
print("每个样本的预测类别:\n", predicted_class_indices)

# 将概率转化为类别
predicted_classes = label_encoder.inverse_transform(predicted_class_indices)
# 打印每个样本的预测类别
print("每个样本的预测类别:\n", predicted_classes)