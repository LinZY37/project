import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, Dropout, Flatten
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
# target = 'alert'  # 假设标签列名为 '标签'
#
# # 获取特征和目标变量
# X = df[features].values
# y = df[target].values
#
# # 标准化特征数据
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
#
# X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
#
# # 将数据分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#
# # 构建CNN + LSTM模型
# model = Sequential()
#
#
# # LSTM层，捕捉长期依赖
# model.add(LSTM(50, activation='relu'))
#
# # Dropout层防止过拟合
# model.add(Dropout(0.2))
#
#
#
# # 全连接层，输出预测值
# model.add(Dense(1, activation='sigmoid'))  # 由于是二分类问题，激活函数用sigmoid
#
# # 编译模型
# model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
#
# # 训练模型
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
#
# # 评估模型
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'测试集上的准确率: {accuracy:.4f}')
#
# # 做出预测
# y_pred = model.predict(X_test)
# y_pred = (y_pred > 0.5).astype(int)  # 将概率转为类别
#
# # 显示预测结果
# print("预测值:", y_pred[:10])
# print("实际值:", y_test[:10])

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam

# 读取训练集数据
train_df = pd.read_excel('1data_zexi.xlsx')

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
target = 'alert'  # 假设标签列名为 '标签'

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

# 调整数据形状为适应LSTM输入
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# 构建CNN + LSTM模型
model = Sequential()

# LSTM层，捕捉长期依赖
model.add(LSTM(50, activation='relu'))

# Dropout层防止过拟合
model.add(Dropout(0.2))

# 全连接层，输出预测值
model.add(Dense(1, activation='sigmoid'))  # 由于是二分类问题，激活函数用sigmoid

# 编译模型
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))

# 评估模型
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'测试集上的准确率: {accuracy:.4f}')

# 做出预测
y_pred = model.predict(X_test_scaled)
y_pred = (y_pred > 0.5).astype(int)  # 将概率转为类别

# 显示预测结果
print("预测值:", y_pred[:200])
print("实际值:", y_test[:200])

