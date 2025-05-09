import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------------
# 1. 数据读取与预处理
# ----------------------
# 读取训练数据（假设CSV文件中包含时间戳和泥水位两列）
train_data = pd.read_csv("m2.csv").head(1500)
mud_level_train = train_data[['mud_level']].values  # 提取训练集泥水位数据

# 读取测试数据（假设CSV文件中包含时间戳和泥水位两列）
test_data = pd.read_csv("hopemud.csv")
mud_level_test = test_data[['mud_level']].values  # 提取测试集泥水位数据

# 数据归一化（仅使用泥水位）
scaler = MinMaxScaler(feature_range=(0, 1))
mud_level_train_scaled = scaler.fit_transform(mud_level_train)
mud_level_test_scaled = scaler.transform(mud_level_test)  # 这里使用transform，保证使用训练集的归一化参数

# ----------------------
# 2. 构建滑动窗口数据集
# ----------------------
def create_dataset(data, window_size=24):
    X, Y = [], []
    for i in range(len(data) - window_size):
        # 输入特征：过去 window_size 个时间步的泥水位
        X.append(data[i:(i + window_size), 0])
        # 输出标签：下一个时间步的泥水位
        Y.append(data[i + window_size, 0])
    return np.array(X), np.array(Y)

window_size = 24  # 根据泥石流数据周期调整（12）
X_train, Y_train = create_dataset(mud_level_train_scaled, window_size)
X_test, Y_test = create_dataset(mud_level_test_scaled, window_size)
print(Y_test)
# 数据集划分（保持时间顺序）
# 训练集和测试集已经从文件中划分了，所以无需额外划分
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# 调整输入形状为LSTM需要的格式 [samples, time_steps, features]
X_train = X_train.reshape(X_train.shape[0], window_size, 1)
X_test = X_test.reshape(X_test.shape[0], window_size, 1)

# ----------------------
# 3. 构建LSTM模型（简化版）
# ----------------------
model = Sequential()
model.add(Bidirectional(LSTM(200, return_sequences=True), input_shape=(window_size, 1)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(200)))
model.add(Dropout(0.3))
model.add(Dense(1))  # 输出层直接预测泥水位值

from tensorflow.keras import losses
# 在模型编译时使用明确的损失函数
model.compile(optimizer='adam', loss=losses.MeanSquaredError())

# ----------------------
# 4. 训练模型
# ----------------------
history = model.fit(
    X_train, Y_train,
    epochs=1,
    batch_size=16,
    validation_data=(X_test, Y_test),
    shuffle=False  # 保持时间序列顺序
)

# ----------------------
# 5. 预测与结果评估
# ----------------------
# 进行预测
y_pred = model.predict(X_test)

# 反归一化
Y_test_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))
y_pred_actual = scaler.inverse_transform(y_pred)
y_pred_actual = np.round(y_pred_actual, 5)

# 计算指标
print(Y_test_actual)
print(y_pred_actual)
mse = mean_squared_error(Y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test_actual, y_pred_actual)
r2 = r2_score(Y_test_actual, y_pred_actual)

print(f'MSE: {mse:.5f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R²: {r2:.2f}')

