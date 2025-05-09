import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from nbeats_tensorflow.model import NBeats
from nbeats_tensorflow.plots import plot

# 读取数据
data1 = pd.read_csv('m2.csv')  # 假设有时间戳列
y_train = data1['mud_level'].values  # 提取泥浆液位数据
y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min())  # 归一化到 [0, 1]

data2 = pd.read_csv('test.csv')  # 假设有时间戳列
y_test = data2['mud_level'].values  # 提取泥浆液位数据
y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min())  # 归一化到 [0, 1]

# # 划分训练集和测试集
# y_train, y_test = train_test_split(y, test_size=0.1, shuffle=False)  # 保证数据按时间顺序划分

# 初始化模型
model = NBeats(
    y=y_train,
    forecast_period=1,
    lookback_period=24,
    units=40,
    num_trend_coefficients=5,
    num_seasonal_coefficients=4,
    num_blocks_per_stack=2,
    share_weights=True,
    share_coefficients=False,
)

# 训练模型
model.fit(
    loss='mse',
    epochs=100,  # 增加训练周期
    batch_size=32,
    learning_rate=0.003,
    backcast_loss_weight=0.5,
    verbose=True
)

# 使用滑动窗口进行预测
res = []
for i in range(len(y_test)-24):
    y_ = y_test[i:i+24]
    df = model.forecast(y=y_, return_backcast=True)
    pro = df['forecast'].dropna().iloc[-1]  # 获取最后一个非NaN预测值
    res.append(pro)

# 实际值
act = y_test[24:]

# 计算误差和R²
mse = mean_squared_error(res, act)
rmse = np.sqrt(mse)
mae = mean_absolute_error(res, act)
r2 = r2_score(res, act)

print(f'MSE: {mse:.5f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R²: {r2:.2f}')



# 计算残差
residuals = act - res

# 绘制箱型图
plt.figure(figsize=(5, 6))
sns.boxplot(data=residuals, boxprops=dict(facecolor='yellow', color='black'))  # 设置箱体颜色为黄色

# 添加标题和标签
plt.title('Residual Boxplot')
plt.ylabel('Residuals')

# 保存图像
file_path = 'residual_boxplot.png'  # 可以修改保存的路径和文件名
plt.savefig(file_path)

# 显示图形
plt.show()

print(f"图像已保存为 {file_path}")