import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# 读取数据集
data = pd.read_csv('data.csv')

# 过滤出标签为1的样本
data_label_1 = data[data['标签'] == 1]

# 使用标签为1的数据训练线性回归模型
features = ['降雨强度', '累积雨量', 'mud_level']
X = data_label_1[features]
y = data_label_1['标签']

# 拟合线性回归模型
model = LinearRegression()
model.fit(X, y)

# 生成新的数据样本（通过调整特征值来产生标签为1的样本）
# 获取标签为0的数据
data_label_0 = data[data['标签'] == 0]

# 尝试微调降雨强度、累积雨量和mud_level的值来生成标签为1的样本
augmented_data = []
for _, row in data_label_0.iterrows():
    original_values = row[features].values
    adjusted_values = original_values.copy()

    # 进行适度的调整（增加降雨强度、累积雨量、mud_level）
    adjusted_values[0] += np.random.uniform(0, 5)  # 调整降雨强度
    adjusted_values[1] += np.random.uniform(0, 10)  # 调整累积雨量
    adjusted_values[2] += np.random.uniform(0, 2)  # 调整mud_level

    # 用线性回归模型预测调整后的数据是否会生成标签为1
    prediction = model.predict([adjusted_values])

    # 如果预测标签为1，则将该样本加入增强数据
    if prediction >= 0.5:
        augmented_data.append(row.tolist() + [1])  # 标签为1的样本

# 创建增强后的数据框
augmented_df = pd.DataFrame(augmented_data, columns=data.columns)

# 合并原始数据和增强的数据
final_data = pd.concat([data, augmented_df], ignore_index=True)

# 打印新的数据集
print(final_data.head())
