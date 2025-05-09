import keras
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Bidirectional, Flatten, TimeDistributed

# 加载数据集
data = pd.read_csv("mud2_with_labels.csv")  # 确保文件是 CSV 格式，使用 pd.read_csv

# 特征缩放
scaler = MinMaxScaler(feature_range=(0,1))

# 选择特征和标签（label 是目标列）
features = ['mud_level']
target = 'label'

# 对特征进行归一化
data[features] = scaler.fit_transform(data[features])

# 准备模型输入的数据
X = []
Y = []
window_size = 50  # 时间窗口大小
for i in range(0, len(data) - window_size - 1, 1):
    temp = []
    for feature in features:
        temp.append(data[feature][i:i+window_size].values)
    X.append(np.array(temp).T)  # 数据形状为 (50, num_features)
    Y.append(data[target][i + window_size])

# 划分训练集和测试集
x_train, x_test, train_label, test_label = train_test_split(X, Y, test_size=0.2, shuffle=True)

train_X = np.array(x_train)
test_X = np.array(x_test)
train_label = np.array(train_label)
test_label = np.array(test_label)

# 重塑数据形状以适应模型输入
train_X = train_X.reshape(train_X.shape[0], 1, window_size, len(features))
test_X = test_X.reshape(test_X.shape[0], 1, window_size, len(features))

# 构建模型

# 创建模型
model = keras.Sequential()

# 直接使用 Conv1D，不需要 TimeDistributed
model.add(Conv1D(128, kernel_size=1, activation='relu', input_shape=(window_size, len(features))))

# 最大池化层
model.add(MaxPooling1D(2))

# 另一个卷积层
model.add(Conv1D(256, kernel_size=1, activation='relu'))
model.add(MaxPooling1D(2))

# 另一个卷积层
model.add(Conv1D(512, kernel_size=1, activation='relu'))
model.add(MaxPooling1D(2))

# 展平层
model.add(Flatten())

# 双向LSTM层
model.add(Bidirectional(LSTM(200, return_sequences=True)))
model.add(Dropout(0.25))

# 另一个双向LSTM层
model.add(Bidirectional(LSTM(200, return_sequences=False)))
model.add(Dropout(0.5))

# 输出层，使用 sigmoid 激活函数
model.add(Dense(1, activation='sigmoid'))  # Sigmoid 用于二分类问题
model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

# 模型训练
model.fit(train_X, train_label, validation_data=(test_X, test_label), epochs=40, batch_size=64, shuffle=False)

# 测试和预测
predicted = model.predict(test_X)
predicted = np.round(predicted).astype(int)  # 将预测结果四舍五入为 0 或 1

# 可视化结果
plt.plot(test_label, label='真实标签', color='blue')
plt.plot(predicted, label='预测标签', color='green')
plt.title('滑坡风险预测')
plt.xlabel('时间')
plt.ylabel('警报（0：无风险，1：有风险）')
plt.legend()
plt.show()
