import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # 导入SMOTE库
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam

# # 读取数据
# df = pd.read_excel('1data_zexi.xlsx')
#
# # 确保时间列是日期时间格式
# df['time'] = pd.to_datetime(df['time'])
#
# # 设置时间戳为索引
# df.set_index('time', inplace=True)
#
# # 特征选择和标签
# features = ['降雨强度', '累积雨量', 'mud_level', '高程', '相对高差', '土壤类型', '纵坡降', '流域面积']
# target = 'alert'  # 假设标签列名为 'alert'
#
# # 获取特征和目标变量
# X = df[features].values
# y = df[target].values
#
# # 标准化特征数据
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # 将数据分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#
# # 使用SMOTE进行过采样处理
# smote = SMOTE(random_state=42)
#
# # 通过SMOTE合成新的样本
# X_res, y_res = smote.fit_resample(X_train, y_train)
#
# # 确认新数据集的类别分布
# print("原始训练集类别分布:", dict(zip(*np.unique(y_train, return_counts=True))))
# print("SMOTE增强后的训练集类别分布:", dict(zip(*np.unique(y_res, return_counts=True))))
#
# # 将数据转换为适合LSTM的形状
# # LSTM要求输入的形状为 (样本数, 时间步长, 特征数)，这里我们只有1个时间步长
#
# X_res = X_res.reshape((X_res.shape[0], 1, X_res.shape[1]))  # 转换为3D
#
# # 将测试集转换为适合LSTM的形状
# X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))  # 转换为3D
#
# # 构建LSTM模型
# model = Sequential()
# model.add(LSTM(50, activation='relu'))
# model.add(Dropout(0.2))  # Dropout层防止过拟合
# model.add(Dense(1, activation='sigmoid'))  # 由于是二分类问题，激活函数用sigmoid
#
# # 编译模型
# model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
#
# # 训练模型
# model.fit(X_res, y_res, epochs=10, batch_size=32, validation_data=(X_test, y_test))
#
# # 评估模型
# y_pred = model.predict(X_test)
# y_pred = (y_pred > 0.5).astype(int)  # 将概率转为类别
#
# # 输出测试集上的准确率
# accuracy = accuracy_score(y_test, y_pred)
# print(f'测试集上的准确率: {accuracy:.4f}')
#
# # 显示部分预测结果
# print("预测值:", y_pred[:10])
# print("实际值:", y_test[:10])


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
import numpy as np

# 读取训练集数据
train_df = pd.read_excel('train.xlsx')

# 读取测试集数据
test_df = pd.read_excel('test.xlsx')

# 确保时间列是日期时间格式
train_df['time'] = pd.to_datetime(train_df['time'])
test_df['time'] = pd.to_datetime(test_df['time'])

# 设置时间戳为索引
train_df.set_index('time', inplace=True)
test_df.set_index('time', inplace=True)

# 特征选择和标签
features = ['降雨强度', '累积雨量', 'mud_level', '高程', '相对高差', '土壤类型', '纵坡降', '流域面积']
target = 'alert'  # 假设标签列名为 'alert'

# 获取训练集特征和目标变量
X_train = train_df[features].values
y_train = train_df[target].values

# 获取测试集特征和目标变量
X_test = test_df[features].values
y_test = test_df[target].values

# 标准化特征数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用SMOTE进行过采样处理
smote = SMOTE(random_state=42)

# 通过SMOTE合成新的样本
X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

# 确认新数据集的类别分布
print("原始训练集类别分布:", dict(zip(*np.unique(y_train, return_counts=True))))
print("SMOTE增强后的训练集类别分布:", dict(zip(*np.unique(y_res, return_counts=True))))

# 将数据转换为适合LSTM的形状
# LSTM要求输入的形状为 (样本数, 时间步长, 特征数)，这里我们只有1个时间步长

X_res = X_res.reshape((X_res.shape[0], 1, X_res.shape[1]))  # 转换为3D

# 将测试集转换为适合LSTM的形状
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))  # 转换为3D

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))  # Dropout层防止过拟合
model.add(Dense(1, activation='sigmoid'))  # 由于是二分类问题，激活函数用sigmoid

# 编译模型
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_res, y_res, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))

# 评估模型
y_pred = model.predict(X_test_scaled)
y_pred = (y_pred > 0.5).astype(int)  # 将概率转为类别

# 输出测试集上的准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'测试集上的准确率: {accuracy:.4f}')

# 显示部分预测结果
print("预测值:", y_pred[:10])
print("实际值:", y_test[:10])
